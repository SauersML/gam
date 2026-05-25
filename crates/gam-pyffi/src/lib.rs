use csv::StringRecord;
use faer::Side;
use gam::basis::create_duchon_basis_1d_derivative_dense;
use gam::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeFitResult, DeviationRuntime, LatentMeasureKind,
};
use gam::estimate::{
    BlockRole, EstimationError, ExternalOptimOptions,
    optimize_external_designwith_heuristic_lambdas, saved_latent_cloglog_state_from_fit,
    saved_mixture_state_from_fit, saved_sas_state_from_fit,
};
use gam::faer_ndarray::{
    FaerCholesky, FaerEigh, array2_to_matmut, factorize_symmetricwith_fallback, fast_xt_diag_x,
};
use gam::families::family_meta::inverse_link_to_binomial_spec;
use gam::families::inverse_link::apply_inverse_link_vec;
use gam::families::scale_design::{build_scale_deviation_transform, infer_non_intercept_start};
use gam::families::survival_construction::{SavedSurvivalTimeBasis, survival_likelihood_modename};
use gam::families::survival_location_scale::{
    ResidualDistribution, residual_distribution_from_inverse_link,
};
use gam::families::survival_predict::{
    apply_inverse_link_state_to_fit_result, fit_result_from_saved_model_for_prediction,
};
use gam::gamlss::{BinomialLocationScaleFitResult, GaussianLocationScaleFitResult};
use gam::gaussian_reml::{
    GaussianRemlMultiBackwardProblem, build_gaussian_reml_eigen_cache_batched,
    gaussian_reml_free_b_score, gaussian_reml_multi_closed_form_backward,
    gaussian_reml_multi_closed_form_backward_batch,
    gaussian_reml_multi_closed_form_backward_from_fit, gaussian_reml_multi_closed_form_with_cache,
};
use gam::geometry::simplex::simplex_frechet_mean;
use gam::hmc::{NutsConfig, NutsResult};
use gam::inference::data::{
    EncodedDataset, UnseenCategoryPolicy, encode_recordswith_inferred_schema,
    encode_recordswith_schema,
};
use gam::inference::formula_dsl::{ParsedTerm, parse_formula, parse_surv_response};
use gam::inference::model::{
    ColumnKindTag, DataSchema, FittedFamily, FittedModel, FittedModelPayload, GroupMetadata,
    MODEL_PAYLOAD_VERSION, ModelKind, PredictModelClass, SavedAnchoredDeviationRuntime,
    SavedDeploymentExtension, SavedLatentZNormalization, SchemaColumn,
    append_deployment_extension_columns,
};
use gam::inference::posterior_bands::{self, PosteriorPredictBandsPayload};
use gam::inference::predict_input::build_predict_input_for_model;
use gam::report::{CoefficientRow, EdfBlockRow, ReportInput, render_html};
use gam::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use gam::survival_marginal_slope::SurvivalMarginalSlopeFitResult;
use gam::terms::basis::{
    BasisOptions, CenterStrategy, Dense, DuchonBasisSpec, DuchonNullspaceOrder, MaternBasisSpec,
    MaternIdentifiability, MaternNu, OneDimensionalBoundary, PeriodicBSplineBasisSpec,
    SpatialIdentifiability, SphereMethod, SphereWahbaKernel, SphericalSplineBasisSpec,
    SplineScratch, auto_centers_1d_equal_mass, auto_knot_vector_1d_quantile,
    bspline_tensor_first_derivative, build_duchon_basis, build_duchon_basis_mixed_periodicity_auto,
    build_duchon_operator_penalty_matrices, build_matern_basis, build_periodic_bspline_basis_1d,
    build_spherical_spline_basis, build_thin_plate_penalty_matrix, create_basis,
    create_difference_penalty_matrix, duchon_polynomial_first_derivative_nd,
    duchon_radial_first_derivative_nd, evaluate_bspline_basis_scalar,
    matern_radial_first_derivative_nd, periodic_bspline_first_derivative_nd, resolve_duchon_orders,
    sphere_first_derivative_nd,
};
use gam::terms::input_loc_derivatives::contract_input_loc_gradient;
use gam::terms::latent_coord::{AuxPriorFamily, aux_prior_targets};
use gam::terms::sae_manifold::{
    AssignmentMode, GumbelTemperatureSchedule, SaeAtomBasisKind, SaeManifoldRho, ScheduleKind,
    term_from_padded_blocks_with_mode,
};
use gam::terms::smooth::BlockwisePenalty;
use gam::terms::skip_transcoder::{
    SkipTranscoderRemlInputs, skip_transcoder_reml_metrics as skip_transcoder_reml_metrics_core,
};
use gam::terms::{
    ARDPenalty as CoreARDPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry,
    BlockOrthogonalityPenalty as CoreBlockOrthogonalityPenalty,
    DifferenceOpKind, GatedSAEDecoder, IBPAssignmentPenalty as CoreIBPAssignmentPenalty,
    IsometryPenalty as CoreIsometryPenalty, IvaeRidgeMeanGauge as IvaeRidgeMeanGaugePenalty,
    JumpReLUPenalty as RustJumpReLUPenalty, MaternBasisGradientTarget,
    MechanismSparsityPenalty as CoreMechanismSparsityPenalty,
    NuclearNormPenalty as CoreNuclearNormPenalty, OrthogonalityPenalty as CoreOrthogonalityPenalty,
    ParametricRowPrecisionPriorPenalty, PenaltyConcavity, PenaltyTier, PsiSlice,
    RowPrecisionPriorPenalty, ScadMcpPenalty as CoreScadMcpPenalty, ScalarWeightSchedule,
    SoftmaxAssignmentSparsityPenalty as CoreSoftmaxAssignmentSparsityPenalty,
    SparsityPenalty as CoreSparsityPenalty,
    StreamingMaternBasisGradientEvaluator,
    TopKActivationPenalty, TotalVariationPenalty as CoreTotalVariationPenalty,
};
use gam::transformation_normal::TransformationNormalFitResult;
use gam::types::{InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily, RhoPrior};
use gam::{FitConfig, FitRequest, FitResult, fit_model, materialize, resolve_offset_column};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayViewD, Axis, IxDyn, s};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArrayDyn,
};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyFloat, PyList, PyString, PyTuple};
use rayon::prelude::*;
use regex::Regex;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

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
    gpu: Option<String>,
    device: Option<String>,

    // Integration seam for task 04's group abstraction. The proposed group
    // type can pass either `group_metadata` directly or `groups` entries with
    // `{name|id|key, metadata}`; fitting ignores it and persistence carries it.
    group_metadata: Option<GroupMetadata>,
    groups: Option<serde_json::Value>,
    precision_hyperpriors: Option<serde_json::Value>,
    penalty_block_gamma_priors: Option<serde_json::Value>,
    latents: Option<serde_json::Value>,
    penalties: Option<serde_json::Value>,
    topology_auto_selector: Option<serde_json::Value>,

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
struct PyExtendGroupRequest {
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    term: Option<String>,
    #[serde(default)]
    column: Option<String>,
    #[serde(default)]
    level: Option<serde_json::Value>,
    #[serde(default)]
    levels: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default)]
    prior: Option<serde_json::Value>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct PyExtensionPrior {
    #[serde(default)]
    mean: Option<f64>,
    #[serde(default)]
    mu: Option<f64>,
    #[serde(default)]
    variance: Option<f64>,
    #[serde(default)]
    precision: Option<f64>,
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
struct PyPredictOptionsPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    interval: Option<f64>,
    #[serde(skip_serializing_if = "bool_is_false")]
    with_uncertainty: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_grid: Option<Vec<f64>>,
}

fn bool_is_false(value: &bool) -> bool {
    !*value
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PySharedPrecisionRequest {
    models: Vec<PySharedPrecisionModel>,
    groups: Vec<PySharedPrecisionGroup>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PySharedPrecisionModel {
    key: serde_json::Value,
    state_json: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PySharedPrecisionGroup {
    name: String,
    shape: f64,
    rate: f64,
    labels: Vec<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    group_metadata: Option<GroupMetadata>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    deployment_extensions: Vec<SavedDeploymentExtension>,
    deviance: f64,
    reml_score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    null_space_logdet: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    null_dim: Option<f64>,
    iterations: usize,
    edf_total: Option<f64>,
    lambdas: Vec<f64>,
    coefficients: Vec<SummaryCoefficientRow>,
    covariance_kind: Option<String>,
    covariance_n: Option<usize>,
    covariance_flat: Option<Vec<f64>>,
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

#[derive(Deserialize)]
struct SurvivalPredictionJsonPayload {
    class: String,
    model_class: Option<String>,
    times: Option<Vec<f64>>,
    hazard: Option<Vec<Vec<f64>>>,
    survival: Option<Vec<Vec<f64>>>,
    cumulative_hazard: Option<Vec<Vec<f64>>>,
    linear_predictor: Option<Vec<f64>>,
    columns: Option<BTreeMap<String, Vec<f64>>>,
    survival_se: Option<Vec<Vec<f64>>>,
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

fn sklearn_response_column_name(formula: &str) -> Option<String> {
    if !formula.contains('~') {
        return None;
    }
    let candidate = formula.split('~').next()?.trim();
    if candidate.is_empty() || candidate.starts_with("Surv(") {
        return None;
    }
    let without_underscores = candidate.replace('_', "");
    if without_underscores.is_empty()
        || !without_underscores
            .chars()
            .all(|character| character.is_alphanumeric())
    {
        return None;
    }
    Some(candidate.to_string())
}

fn sklearn_resolved_formula(formula: &str, target_name: &str) -> String {
    match formula.split_once('~') {
        Some((_lhs, rhs)) => format!("{target_name} ~ {}", rhs.trim()),
        None => format!("{target_name} ~ {}", formula.trim()),
    }
}

#[pyfunction(signature = (columns, formula, target_column = None, has_external_target = false))]
fn sklearn_fit_metadata(
    columns: Vec<String>,
    formula: &str,
    target_column: Option<String>,
    has_external_target: bool,
) -> PyResult<(String, Vec<String>, String)> {
    let has_target_column = target_column.is_some();
    if has_target_column && has_external_target {
        return Err(py_value_error(
            "target_column and has_external_target are mutually exclusive".to_string(),
        ));
    }

    let target_name = if let Some(target_column) = target_column {
        if !columns.iter().any(|column| column == &target_column) {
            return Err(py_value_error(format!(
                "target column '{target_column}' is missing from the training table"
            )));
        }
        target_column
    } else if has_external_target {
        sklearn_response_column_name(formula).unwrap_or_else(|| "y".to_string())
    } else {
        let target_name = sklearn_response_column_name(formula).ok_or_else(|| {
            py_value_error("formula must include a response when y is not provided".to_string())
        })?;
        if !columns.iter().any(|column| column == &target_name) {
            return Err(py_value_error(format!(
                "response column '{target_name}' is missing from the training table"
            )));
        }
        target_name
    };

    if has_external_target && columns.iter().any(|column| column == &target_name) {
        return Err(py_value_error(format!(
            "target column '{target_name}' already exists in the feature table"
        )));
    }

    let fit_formula = if has_external_target || has_target_column {
        sklearn_resolved_formula(formula, &target_name)
    } else {
        formula.to_string()
    };
    let feature_names = columns
        .into_iter()
        .filter(|column| column != &target_name)
        .collect();

    Ok((fit_formula, feature_names, target_name))
}

enum PythonExceptionKind {
    Formula,
    SchemaMismatch,
    Prediction,
    Gam,
}

impl PythonExceptionKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Formula => "formula",
            Self::SchemaMismatch => "schema_mismatch",
            Self::Prediction => "prediction",
            Self::Gam => "gam",
        }
    }
}

#[pyfunction]
fn classify_exception_message(message: String) -> &'static str {
    let lower = message.to_lowercase();
    let kind = if lower.contains("formula") || lower.contains("parse") {
        PythonExceptionKind::Formula
    } else if lower.contains("schema")
        || lower.contains("missing required column")
        || lower.contains("unknown column")
    {
        PythonExceptionKind::SchemaMismatch
    } else if lower.contains("prediction") || lower.contains("predict") {
        PythonExceptionKind::Prediction
    } else {
        PythonExceptionKind::Gam
    };
    kind.as_str()
}

#[pyfunction]
fn torch_from_fitted(
    module_cls: &Bound<'_, PyAny>,
    model: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if !model.hasattr("_model_bytes")? {
        return Err(py_value_error(
            "from_fitted requires a fitted gamfit.Model instance".to_string(),
        ));
    }
    let module = module_cls.call0()?;
    module.setattr("_model", model)?;
    Ok(module.unbind())
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
            "extend_with_group",
            "torch_from_fitted",
            "predict",
            "predict_array",
            "build_predict_payload_json",
            "interpolate_survival_surface",
            "survival_chunk_iter_collect",
            "write_survival_csv",
            "default_survival_time_grid",
            "competing_risks_cif",
            "competing_risks_cif_from_predictions",
            "competing_risks_prediction_payload_from_json",
            "sample",
            "summary",
            "summary_html",
            "check",
            "report",
            "sklearn_fit_metadata",
            "classify_exception_message",
            "cross_fit_shared_precision_groups",
            "save",
            "validate_formula",
            "formula_validation_repr",
            "formula_validation_html",
            "design_matrix_array",
            "basis",
            "basis_with_jet",
            "duchon_function_norm_penalty",
            "duchon_operator_penalties",
            "thin_plate_penalty",
            "gaussian_weighted_ridge_array",
            "gaussian_weighted_ridge_batch",
            "gaussian_reml_score",
            "gaussian_reml_fit",
            "gaussian_reml_fit_backward",
            "gaussian_reml_fit_batched",
            "gaussian_reml_fit_batched_backward",
            "gaussian_reml_fit_positions",
            "gaussian_reml_fit_positions_backward",
            "gaussian_reml_fit_positions_batched",
            "gaussian_reml_fit_positions_batched_backward",
            "gaussian_reml_fit_latent",
            "gaussian_reml_fit_latent_backward",
            "glm_reml_fit_latent",
            "glm_reml_fit_latent_backward",
            "equivariant_penalty_value",
            "skip_transcoder_reml_metrics",
            "skip_transcoder_select_reml",
            "_block_diag",
            "tierney_kadane_normalized_score",
            "gaussian_reml_fit_formula_table",
            "gaussian_reml_fit_with_constraints_forward",
            "gaussian_reml_fit_with_constraints_backward",
            "sphere_frechet_mean",
            "response_geometry_closure",
            "response_geometry_clr",
            "response_geometry_alr",
            "response_geometry_inverse_alr",
            "response_geometry_simplex_frechet_mean",
            "response_geometry_simplex_log_map",
            "response_geometry_simplex_exp_map",
            "response_geometry_sphere_log_map",
            "response_geometry_sphere_exp_map",
            "response_geometry_normalize_fisher_rao",
            "equivariant_rho_so2",
            "equivariant_rho_so2_jvp",
            "equivariant_rho_so3",
            "equivariant_rho_so3_jvp",
            "equivariant_gauge_companion_loss",
        ],
    )?;
    info.set_item(
        "supported_model_classes",
        vec![
            "standard",
            "transformation-normal",
            "survival",
            "competing-risks-survival",
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
fn interpolate_survival_surface<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    query: PyReadonlyArray1<'py, f64>,
    clip_lo: Option<f64>,
    clip_hi: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let grid_view = grid.as_array();
    let surface_view = surface.as_array();
    let query_values: Vec<f64> = query.as_array().iter().copied().collect();
    let n_grid = grid_view.len();
    let (n_rows, n_cols) = surface_view.dim();
    if n_grid == 0 || n_cols != n_grid {
        return Err(py_value_error(
            "survival interpolation requires a non-empty grid".to_string(),
        ));
    }

    let mut order: Vec<(f64, usize)> = grid_view
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let sorted_grid: Vec<f64> = order.iter().map(|(value, _index)| *value).collect();
    let sorted_indices: Vec<usize> = order
        .iter()
        .map(|(_value, index)| *index)
        .collect();
    let n_query = query_values.len();
    let mut values = vec![0.0_f64; n_rows * n_query];

    values
        .par_chunks_mut(n_query.max(1))
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            if n_query == 0 {
                return;
            }
            let surface_row = surface_view.row(row_idx);
            for (query_idx, query_value) in query_values.iter().copied().enumerate() {
                let mut interpolated = if query_value.is_nan() {
                    f64::NAN
                } else if query_value <= sorted_grid[0] {
                    surface_row[sorted_indices[0]]
                } else if query_value >= sorted_grid[n_grid - 1] {
                    surface_row[sorted_indices[n_grid - 1]]
                } else {
                    let upper =
                        sorted_grid.partition_point(|grid_value| *grid_value <= query_value);
                    let lower = upper - 1;
                    let x0 = sorted_grid[lower];
                    let x1 = sorted_grid[upper];
                    let y0 = surface_row[sorted_indices[lower]];
                    let y1 = surface_row[sorted_indices[upper]];
                    if x1 == x0 {
                        y1
                    } else {
                        y0 + (query_value - x0) * (y1 - y0) / (x1 - x0)
                    }
                };
                if let Some(lo) = clip_lo {
                    if interpolated < lo {
                        interpolated = lo;
                    }
                }
                if let Some(hi) = clip_hi {
                    if interpolated > hi {
                        interpolated = hi;
                    }
                }
                out_row[query_idx] = interpolated;
            }
        });

    let out = Array2::from_shape_vec((n_rows, n_query), values).map_err(|err| {
        py_value_error(format!(
            "failed to assemble survival interpolation result: {err}"
        ))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn interpolate_rows<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    query: PyReadonlyArray1<'py, f64>,
    clip_lo: Option<f64>,
    clip_hi: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let grid_view = grid.as_array();
    let surface_view = surface.as_array();
    let query_values: Vec<f64> = query.as_array().iter().copied().collect();
    let n_grid = grid_view.len();
    let (n_rows, n_cols) = surface_view.dim();
    if n_grid == 0 || n_cols != n_grid {
        return Err(py_value_error(
            "survival interpolation requires a non-empty grid".to_string(),
        ));
    }

    let mut order: Vec<(f64, usize)> = grid_view
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let sorted_grid: Vec<f64> = order.iter().map(|(value, _index)| *value).collect();
    let sorted_indices: Vec<usize> = order.iter().map(|(_value, index)| *index).collect();
    let n_query = query_values.len();
    let mut values = vec![0.0_f64; n_rows * n_query];

    values
        .par_chunks_mut(n_query.max(1))
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            if n_query == 0 {
                return;
            }
            let surface_row = surface_view.row(row_idx);
            for (query_idx, query_value) in query_values.iter().copied().enumerate() {
                let mut interpolated = if query_value.is_nan() {
                    f64::NAN
                } else if query_value <= sorted_grid[0] {
                    surface_row[sorted_indices[0]]
                } else if query_value >= sorted_grid[n_grid - 1] {
                    surface_row[sorted_indices[n_grid - 1]]
                } else {
                    let upper =
                        sorted_grid.partition_point(|grid_value| *grid_value <= query_value);
                    let lower = upper - 1;
                    let x0 = sorted_grid[lower];
                    let x1 = sorted_grid[upper];
                    let y0 = surface_row[sorted_indices[lower]];
                    let y1 = surface_row[sorted_indices[upper]];
                    if x1 == x0 {
                        y1
                    } else {
                        y0 + (query_value - x0) * (y1 - y0) / (x1 - x0)
                    }
                };
                if let Some(lo) = clip_lo {
                    if interpolated < lo {
                        interpolated = lo;
                    }
                }
                if let Some(hi) = clip_hi {
                    if interpolated > hi {
                        interpolated = hi;
                    }
                }
                out_row[query_idx] = interpolated;
            }
        });

    let out = Array2::from_shape_vec((n_rows, n_query), values)
        .map_err(|err| py_value_error(format!("failed to assemble interpolation result: {err}")))?;
    Ok(out.into_pyarray(py))
}

fn write_csv_field(writer: &mut BufWriter<File>, value: &str) -> std::io::Result<()> {
    let needs_quotes = value
        .bytes()
        .any(|byte| matches!(byte, b',' | b'"' | b'\n' | b'\r'));
    if !needs_quotes {
        writer.write_all(value.as_bytes())?;
        return Ok(());
    }
    writer.write_all(b"\"")?;
    for byte in value.bytes() {
        if byte == b'"' {
            writer.write_all(b"\"\"")?;
        } else {
            writer.write_all(&[byte])?;
        }
    }
    writer.write_all(b"\"")
}

fn survival_csv_interpolate(
    surface_values: &[f64],
    n_cols: usize,
    sorted_grid: &[f64],
    sorted_indices: &[usize],
    row_idx: usize,
    query_value: f64,
) -> f64 {
    if query_value.is_nan() {
        return f64::NAN;
    }
    let n_grid = sorted_grid.len();
    let row_offset = row_idx * n_cols;
    if query_value <= sorted_grid[0] {
        surface_values[row_offset + sorted_indices[0]]
    } else if query_value >= sorted_grid[n_grid - 1] {
        surface_values[row_offset + sorted_indices[n_grid - 1]]
    } else {
        let upper = sorted_grid.partition_point(|grid_value| *grid_value <= query_value);
        let lower = upper - 1;
        let x0 = sorted_grid[lower];
        let x1 = sorted_grid[upper];
        let y0 = surface_values[row_offset + sorted_indices[lower]];
        let y1 = surface_values[row_offset + sorted_indices[upper]];
        if x1 == x0 {
            y1
        } else {
            y0 + (query_value - x0) * (y1 - y0) / (x1 - x0)
        }
    }
}

fn clip_survival_surface_value(mut value: f64, clip_lo: Option<f64>, clip_hi: Option<f64>) -> f64 {
    if let Some(lo) = clip_lo {
        if value < lo {
            value = lo;
        }
    }
    if let Some(hi) = clip_hi {
        if value > hi {
            value = hi;
        }
    }
    value
}

#[pyfunction]
fn survival_chunk_iter_collect<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    kind: &str,
    clip_lo: Option<f64>,
    clip_hi: Option<f64>,
    people_chunk: usize,
    time_grid_chunk: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    match kind {
        "hazard" | "cumulative_hazard" | "survival" | "survival_se" => {}
        other => {
            return Err(py_value_error(format!(
                "unknown survival surface kind '{other}'"
            )));
        }
    }
    if people_chunk == 0 {
        return Err(py_value_error("people_chunk must be positive".to_string()));
    }
    if time_grid_chunk == 0 {
        return Err(py_value_error("time_grid_chunk must be positive".to_string()));
    }
    let grid_view = grid.as_array();
    let surface_view = surface.as_array();
    let n_grid = grid_view.len();
    let (n_rows, n_cols) = surface_view.dim();
    if n_grid == 0 || n_cols != n_grid {
        return Err(py_value_error(
            "survival interpolation requires a non-empty grid".to_string(),
        ));
    }

    let mut order: Vec<(f64, usize)> = grid_view
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let sorted_grid: Vec<f64> = order.iter().map(|(value, _index)| *value).collect();
    let sorted_indices: Vec<usize> = order.iter().map(|(_value, index)| *index).collect();
    let mut surface_values = Vec::with_capacity(n_rows * n_cols);
    for row_idx in 0..n_rows {
        for col_idx in 0..n_cols {
            surface_values.push(surface_view[[row_idx, col_idx]]);
        }
    }
    let times_values: Vec<f64> = times.as_array().iter().copied().collect();
    let n_times = times_values.len();

    let values = py.detach(move || {
        let mut values = vec![0.0_f64; n_rows * n_times];
        for row_start in (0..n_rows).step_by(people_chunk) {
            let row_stop = (row_start + people_chunk).min(n_rows);
            for time_start in (0..n_times).step_by(time_grid_chunk) {
                let time_stop = (time_start + time_grid_chunk).min(n_times);
                for row_idx in row_start..row_stop {
                    let out_row_start = row_idx * n_times;
                    for time_idx in time_start..time_stop {
                        let interpolated = survival_csv_interpolate(
                            &surface_values,
                            n_cols,
                            &sorted_grid,
                            &sorted_indices,
                            row_idx,
                            times_values[time_idx],
                        );
                        values[out_row_start + time_idx] =
                            clip_survival_surface_value(interpolated, clip_lo, clip_hi);
                    }
                }
            }
        }
        values
    });
    let out = Array2::from_shape_vec((n_rows, n_times), values).map_err(|err| {
        py_value_error(format!(
            "failed to assemble survival chunk result: {err}"
        ))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn write_survival_csv(
    py: Python<'_>,
    path: &str,
    grid: PyReadonlyArray1<'_, f64>,
    surface: PyReadonlyArray2<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    id_column: Option<String>,
    row_ids: Option<Vec<String>>,
    people_chunk: usize,
    time_grid_chunk: usize,
) -> PyResult<String> {
    if people_chunk == 0 {
        return Err(py_value_error("people_chunk must be positive".to_string()));
    }
    if time_grid_chunk == 0 {
        return Err(py_value_error("time_grid_chunk must be positive".to_string()));
    }
    let grid_view = grid.as_array();
    let surface_view = surface.as_array();
    let n_grid = grid_view.len();
    let (n_rows, n_cols) = surface_view.dim();
    if n_grid == 0 || n_cols != n_grid {
        return Err(py_value_error(
            "survival interpolation requires a non-empty grid".to_string(),
        ));
    }
    if id_column.is_some() {
        match row_ids.as_ref() {
            Some(ids) if ids.len() >= n_rows => {}
            Some(ids) => {
                return Err(py_value_error(format!(
                    "row_ids length {} is smaller than survival row count {n_rows}",
                    ids.len()
                )));
            }
            None => {
                return Err(py_value_error(
                    "row_ids are required when id_column is set".to_string(),
                ));
            }
        }
    }

    let mut order: Vec<(f64, usize)> = grid_view
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let sorted_grid: Vec<f64> = order.iter().map(|(value, _index)| *value).collect();
    let sorted_indices: Vec<usize> = order.iter().map(|(_value, index)| *index).collect();
    let mut surface_values = Vec::with_capacity(n_rows * n_cols);
    for row_idx in 0..n_rows {
        for col_idx in 0..n_cols {
            surface_values.push(surface_view[[row_idx, col_idx]]);
        }
    }
    let times_values: Vec<f64> = times.as_array().iter().copied().collect();
    let path_owned = path.to_string();

    Ok(py.detach(move || -> std::io::Result<String> {
        let file = File::create(&path_owned)?;
        let mut writer = BufWriter::new(file);
        match id_column.as_ref() {
            Some(column) => {
                writer.write_all(b"row,")?;
                write_csv_field(&mut writer, column)?;
                writer.write_all(b",time,survival\n")?;
            }
            None => writer.write_all(b"row,time,survival\n")?,
        }

        for row_start in (0..n_rows).step_by(people_chunk) {
            let row_stop = (row_start + people_chunk).min(n_rows);
            for time_start in (0..times_values.len()).step_by(time_grid_chunk) {
                let time_stop = (time_start + time_grid_chunk).min(times_values.len());
                for row_idx in row_start..row_stop {
                    for query_value in &times_values[time_start..time_stop] {
                        let survival = survival_csv_interpolate(
                            &surface_values,
                            n_cols,
                            &sorted_grid,
                            &sorted_indices,
                            row_idx,
                            *query_value,
                        )
                        .clamp(0.0, 1.0);
                        match (id_column.as_ref(), row_ids.as_ref()) {
                            (Some(_column), Some(ids)) => {
                                write!(writer, "{row_idx},")?;
                                write_csv_field(&mut writer, &ids[row_idx])?;
                                writeln!(writer, ",{query_value},{survival}")?;
                            }
                            _ => writeln!(writer, "{row_idx},{query_value},{survival}")?,
                        }
                    }
                }
            }
        }
        writer.flush()?;
        Ok(path_owned)
    })?)
}

#[pyfunction]
fn survival_coerce_times<'py>(
    py: Python<'py>,
    times: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let values: Vec<f64> = times.as_array().iter().copied().collect();
    if values.is_empty() {
        return Err(py_value_error(
            "survival prediction requires at least one time".to_string(),
        ));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(py_value_error(
            "survival prediction times must be finite".to_string(),
        ));
    }
    Ok(Array1::from_vec(values).into_pyarray(py).unbind())
}

#[pyfunction]
fn survival_parameters_matrix<'py>(
    py: Python<'py>,
    parameters: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let view = parameters.as_array();
    match view.ndim() {
        1 => {
            let n = view.shape()[0];
            let data: Vec<f64> = view.iter().copied().collect();
            let out = Array2::from_shape_vec((n, 1), data).map_err(|err| {
                py_value_error(format!("failed to reshape parameters: {err}"))
            })?;
            Ok(out.into_pyarray(py).unbind())
        }
        2 => {
            let (rows, cols) = (view.shape()[0], view.shape()[1]);
            let data: Vec<f64> = view.iter().copied().collect();
            let out = Array2::from_shape_vec((rows, cols), data).map_err(|err| {
                py_value_error(format!("failed to reshape parameters: {err}"))
            })?;
            Ok(out.into_pyarray(py).unbind())
        }
        other => Err(py_value_error(format!(
            "survival parameters must be 1D or 2D, got {other}D"
        ))),
    }
}

#[pyfunction]
fn survival_collect_chunks<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_times: usize,
    blocks: Vec<(usize, usize, usize, usize, PyReadonlyArray2<'py, f64>)>,
) -> PyResult<Py<PyArray2<f64>>> {
    let mut dense = Array2::<f64>::zeros((n_rows, n_times));
    for (row_start, row_stop, time_start, time_stop, block) in blocks {
        if row_stop > n_rows || time_stop > n_times {
            return Err(py_value_error(
                "survival chunk block exceeds dense matrix bounds".to_string(),
            ));
        }
        let block_view = block.as_array();
        let (br, bc) = (block_view.shape()[0], block_view.shape()[1]);
        if br != row_stop - row_start || bc != time_stop - time_start {
            return Err(py_value_error(
                "survival chunk block shape mismatch".to_string(),
            ));
        }
        for i in 0..br {
            for j in 0..bc {
                dense[[row_start + i, time_start + j]] = block_view[[i, j]];
            }
        }
    }
    Ok(dense.into_pyarray(py).unbind())
}

#[pyfunction]
fn hazard_from_cumulative<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    cumulative: PyReadonlyArray2<'py, f64>,
    previous_cumulative: Option<PyReadonlyArray2<'py, f64>>,
    previous_time: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cum_view = cumulative.as_array();
    let times_view = times.as_array();
    let (n_rows, n_times) = cum_view.dim();
    if times_view.len() != n_times {
        return Err(py_value_error(
            "hazard_from_cumulative requires matching time count".to_string(),
        ));
    }
    let previous = match previous_cumulative {
        Some(arr) => {
            let view = arr.as_array();
            if view.dim() != (n_rows, 1) {
                return Err(py_value_error(
                    "previous_cumulative must have one column per row".to_string(),
                ));
            }
            view.to_owned()
        }
        None => Array2::<f64>::zeros((n_rows, 1)),
    };
    let mut out = Array2::<f64>::zeros((n_rows, n_times));
    for j in 0..n_times {
        let prev_t = if j == 0 {
            previous_time
        } else {
            times_view[j - 1]
        };
        let mut width = times_view[j] - prev_t;
        if width <= 0.0 {
            width = 1.0;
        }
        for i in 0..n_rows {
            let prev_h = if j == 0 {
                previous[[i, 0]]
            } else {
                cum_view[[i, j - 1]]
            };
            out[[i, j]] = (cum_view[[i, j]] - prev_h) / width;
        }
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn survival_cumulative_from_survival<'py>(
    py: Python<'py>,
    survival: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let view = survival.as_array();
    let (n_rows, n_cols) = (view.shape()[0], view.shape()[1]);
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for i in 0..n_rows {
        for j in 0..n_cols {
            let s = view[[i, j]].clamp(1e-12, 1.0);
            out[[i, j]] = -s.ln();
        }
    }
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn survival_block<'py>(
    py: Python<'py>,
    params: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let params = params.as_array();
    let times_view = times.as_array();
    let n_rows = params.shape()[0];
    let n_times = times_view.len();
    if params.shape()[1] == 0 {
        return Err(py_value_error(
            "survival parameter matrix must have at least one column".to_string(),
        ));
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_times));
    for i in 0..n_rows {
        let hazard = params[[i, 0]].exp();
        for j in 0..n_times {
            out[[i, j]] = (-hazard * times_view[j]).exp();
        }
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn survival_ffi_surface<'py>(
    py: Python<'py>,
    times: PyReadonlyArrayDyn<'py, f64>,
    surface: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Option<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)>> {
    let times_view = times.as_array();
    let grid: Vec<f64> = times_view.iter().copied().collect();
    if grid.is_empty() {
        return Ok(None);
    }
    let surf_view = surface.as_array();
    let (n_rows, n_cols) = match surf_view.ndim() {
        1 => (surf_view.shape()[0], 1usize),
        2 => (surf_view.shape()[0], surf_view.shape()[1]),
        _ => return Ok(None),
    };
    if n_cols != grid.len() {
        return Ok(None);
    }
    let surface_data: Vec<f64> = surf_view.iter().copied().collect();
    let surface_arr = Array2::from_shape_vec((n_rows, n_cols), surface_data).map_err(|err| {
        py_value_error(format!("failed to reshape surface: {err}"))
    })?;
    let grid_arr = Array1::from_vec(grid);
    Ok(Some((
        grid_arr.into_pyarray(py).unbind(),
        surface_arr.into_pyarray(py).unbind(),
    )))
}

#[pyfunction]
fn numeric_matrix_validate<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    label: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let np = py.import("numpy")?;
    let mut array = np.getattr("asarray")?.call1((values,))?;
    let ndim = array.getattr("ndim")?.extract::<usize>()?;
    if ndim == 1 {
        array = array.call_method1("reshape", (-1, 1))?;
    } else if ndim != 2 {
        return Err(py_value_error(format!(
            "{label} must be a 1D or 2D numeric array"
        )));
    }

    let (rows, cols) = array.getattr("shape")?.extract::<(usize, usize)>()?;
    if rows == 0 || cols == 0 {
        return Err(py_value_error(format!("{label} cannot be empty")));
    }

    let dtype_value = array.getattr("dtype")?;
    let float64_value = np.getattr("float64")?;
    if !dtype_value.eq(float64_value)? {
        return Err(PyTypeError::new_err(format!(
            "{label} must be a float64 numpy array for zero-copy FFI"
        )));
    }

    {
        let typed_array = array.cast::<PyArray2<f64>>()?;
        let readonly = typed_array.readonly();
        if !readonly.as_array().iter().all(|value| value.is_finite()) {
            return Err(py_value_error(format!(
                "{label} must contain only finite values"
            )));
        }
    }

    array.cast_into::<PyArray2<f64>>()
}

#[pyfunction]
fn marginal_slope_clip_probabilities(values: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(values
        .into_iter()
        .map(|value| value.clamp(0.0, 1.0))
        .collect())
}

#[pyfunction]
fn transformation_normal_z_from_columns(columns_json: &str) -> PyResult<Vec<f64>> {
    let columns: BTreeMap<String, Vec<f64>> = serde_json::from_str(columns_json)
        .map_err(|err| py_value_error(format!("invalid prediction columns json: {err}")))?;
    for key in ["z", "z_score", "transformed", "eta", "mean"] {
        if let Some(values) = columns.get(key) {
            return Ok(values.clone());
        }
    }
    Err(PyKeyError::new_err(
        "transformation-normal prediction payload is missing a z-score column",
    ))
}

#[pyfunction]
fn column_stack_f64<'py>(
    py: Python<'py>,
    columns: Vec<Vec<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    if columns.is_empty() {
        let out = Array2::<f64>::zeros((0, 0));
        return Ok(out.into_pyarray(py).unbind());
    }
    let n_rows = columns[0].len();
    for (idx, col) in columns.iter().enumerate() {
        if col.len() != n_rows {
            return Err(py_value_error(format!(
                "column {idx} has length {} but expected {n_rows}",
                col.len()
            )));
        }
    }
    let n_cols = columns.len();
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for (col_idx, col) in columns.iter().enumerate() {
        for (row_idx, value) in col.iter().enumerate() {
            out[[row_idx, col_idx]] = *value;
        }
    }
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn flat_to_matrix_f64<'py>(
    py: Python<'py>,
    flat: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    if flat.len() != n_rows * n_cols {
        return Err(py_value_error(format!(
            "design matrix FFI payload shape mismatch: got {} floats, expected {} * {}",
            flat.len(),
            n_rows,
            n_cols
        )));
    }
    let out = Array2::from_shape_vec((n_rows, n_cols), flat).map_err(|err| {
        py_value_error(format!("failed to reshape design matrix: {err}"))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn vec_to_array1_f64<'py>(py: Python<'py>, values: Vec<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Ok(Array1::from_vec(values).into_pyarray(py).unbind())
}

#[pyfunction]
fn competing_risks_prediction_payload_from_json(
    py: Python<'_>,
    raw: &str,
) -> PyResult<PyObject> {
    let payload: serde_json::Value = serde_json::from_str(raw)
        .map_err(|err| py_value_error(format!("invalid competing-risks prediction JSON: {err}")))?;
    let object = payload.as_object().ok_or_else(|| {
        py_value_error("competing-risks prediction payload must be a JSON object".to_string())
    })?;
    match object.get("class").and_then(serde_json::Value::as_str) {
        Some("competing_risks_prediction") => {}
        Some(other) => {
            return Err(py_value_error(format!(
                "expected competing_risks_prediction payload, got {other}"
            )));
        }
        None => {
            return Err(py_value_error(
                "competing-risks prediction payload is missing class".to_string(),
            ));
        }
    }

    let out = PyDict::new(py);
    out.set_item(
        "model_class",
        object
            .get("model_class")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("competing risks survival"),
    )?;
    out.set_item(
        "likelihood_mode",
        object
            .get("likelihood_mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
    )?;
    out.set_item(
        "endpoint_names",
        competing_risks_string_list(object.get("endpoint_names"), "endpoint_names")?,
    )?;
    out.set_item(
        "times",
        Array1::from_vec(competing_risks_numeric_list(object.get("times"), "times")?)
            .into_pyarray(py),
    )?;
    set_optional_competing_risks_matrix(py, &out, "hazard", object.get("hazard"))?;
    set_optional_competing_risks_matrix(py, &out, "survival", object.get("survival"))?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "cumulative_hazard",
        object.get("cumulative_hazard"),
    )?;
    set_optional_competing_risks_matrix(py, &out, "cif", object.get("cif"))?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "overall_survival",
        object.get("overall_survival"),
    )?;
    set_optional_competing_risks_vector(
        py,
        &out,
        "linear_predictor",
        object.get("linear_predictor"),
    )?;

    let columns = PyDict::new(py);
    for (name, values) in competing_risks_columns(object.get("columns"))? {
        columns.set_item(name, values)?;
    }
    out.set_item("columns", columns)?;
    Ok(out.into_any().unbind())
}

fn set_optional_competing_risks_vector<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    key: &str,
    raw: Option<&serde_json::Value>,
) -> PyResult<()> {
    match raw {
        Some(serde_json::Value::Null) | None => out.set_item(key, py.None()),
        Some(value) => out.set_item(
            key,
            Array1::from_vec(competing_risks_flattened_numbers(value, key)?).into_pyarray(py),
        ),
    }
}

fn set_optional_competing_risks_matrix<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    key: &str,
    raw: Option<&serde_json::Value>,
) -> PyResult<()> {
    match raw {
        Some(serde_json::Value::Null) | None => out.set_item(key, py.None()),
        Some(value) => out.set_item(
            key,
            competing_risks_numeric_matrix(value, key)?.into_pyarray(py),
        ),
    }
}

fn competing_risks_columns(
    raw: Option<&serde_json::Value>,
) -> PyResult<BTreeMap<String, Vec<f64>>> {
    let Some(value) = raw else {
        return Ok(BTreeMap::new());
    };
    if value.is_null() {
        return Ok(BTreeMap::new());
    }
    let object = value
        .as_object()
        .ok_or_else(|| py_value_error("columns must be a JSON object".to_string()))?;
    let mut columns = BTreeMap::new();
    for (name, values) in object {
        columns.insert(
            name.clone(),
            competing_risks_numeric_list(Some(values), &format!("columns.{name}"))?,
        );
    }
    Ok(columns)
}

fn competing_risks_string_list(
    raw: Option<&serde_json::Value>,
    key: &str,
) -> PyResult<Vec<String>> {
    let Some(value) = raw else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let items = value
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key} must be a JSON array")))?;
    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let text = item
            .as_str()
            .ok_or_else(|| py_value_error(format!("{key}[{idx}] must be a string")))?;
        out.push(text.to_string());
    }
    Ok(out)
}

fn competing_risks_numeric_list(
    raw: Option<&serde_json::Value>,
    key: &str,
) -> PyResult<Vec<f64>> {
    let Some(value) = raw else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let items = value
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key} must be a JSON array")))?;
    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        out.push(competing_risks_number(item, &format!("{key}[{idx}]"))?);
    }
    Ok(out)
}

fn competing_risks_flattened_numbers(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Vec<f64>> {
    let mut out = Vec::new();
    competing_risks_flatten_numbers_into(value, key, &mut out)?;
    Ok(out)
}

fn competing_risks_flatten_numbers_into(
    value: &serde_json::Value,
    key: &str,
    out: &mut Vec<f64>,
) -> PyResult<()> {
    match value {
        serde_json::Value::Number(_) => {
            out.push(competing_risks_number(value, key)?);
            Ok(())
        }
        serde_json::Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                competing_risks_flatten_numbers_into(item, &format!("{key}[{idx}]"), out)?;
            }
            Ok(())
        }
        _ => Err(py_value_error(format!("{key} must contain only numbers"))),
    }
}

fn competing_risks_numeric_matrix(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Array2<f64>> {
    match competing_risks_array_depth(value, key)? {
        1 => competing_risks_vector_as_matrix(value, key),
        2 => competing_risks_matrix2(value, key),
        3 => competing_risks_stacked_matrix3(value, key),
        depth => Err(py_value_error(format!(
            "{key} must be a vector, matrix, or list of matrices; got array depth {depth}"
        ))),
    }
}

fn competing_risks_vector_as_matrix(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Array2<f64>> {
    let values = competing_risks_numeric_list(Some(value), key)?;
    let n_rows = values.len();
    Array2::from_shape_vec((n_rows, 1), values)
        .map_err(|err| py_value_error(format!("failed to reshape {key}: {err}")))
}

fn competing_risks_matrix2(value: &serde_json::Value, key: &str) -> PyResult<Array2<f64>> {
    let rows = value
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key} must be a JSON array")))?;
    if rows.is_empty() {
        return Ok(Array2::<f64>::zeros((0, 1)));
    }
    let first_row = rows[0]
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key}[0] must be a JSON array")))?;
    let n_cols = first_row.len();
    let mut flat = Vec::with_capacity(rows.len() * n_cols);
    for (row_idx, row) in rows.iter().enumerate() {
        let cells = row
            .as_array()
            .ok_or_else(|| py_value_error(format!("{key}[{row_idx}] must be a JSON array")))?;
        if cells.len() != n_cols {
            return Err(py_value_error(format!(
                "{key}[{row_idx}] has length {}, expected {n_cols}",
                cells.len()
            )));
        }
        for (col_idx, cell) in cells.iter().enumerate() {
            flat.push(competing_risks_number(
                cell,
                &format!("{key}[{row_idx}][{col_idx}]"),
            )?);
        }
    }
    Array2::from_shape_vec((rows.len(), n_cols), flat)
        .map_err(|err| py_value_error(format!("failed to reshape {key}: {err}")))
}

fn competing_risks_stacked_matrix3(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Array2<f64>> {
    let matrices = value
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key} must be a JSON array")))?;
    if matrices.is_empty() {
        return Ok(Array2::<f64>::zeros((0, 1)));
    }
    let mut n_cols = None;
    let mut n_rows = 0usize;
    let mut flat = Vec::new();
    for (matrix_idx, matrix_value) in matrices.iter().enumerate() {
        let matrix_key = format!("{key}[{matrix_idx}]");
        let matrix = competing_risks_matrix2(matrix_value, &matrix_key)?;
        match n_cols {
            Some(expected) if matrix.ncols() != expected => {
                return Err(py_value_error(format!(
                    "{matrix_key} has {} columns, expected {expected}",
                    matrix.ncols()
                )));
            }
            Some(_) => {}
            None => {
                n_cols = Some(matrix.ncols());
            }
        }
        n_rows += matrix.nrows();
        flat.extend(matrix.iter().copied());
    }
    let cols = n_cols.unwrap_or(1);
    Array2::from_shape_vec((n_rows, cols), flat)
        .map_err(|err| py_value_error(format!("failed to reshape {key}: {err}")))
}

fn competing_risks_array_depth(value: &serde_json::Value, key: &str) -> PyResult<usize> {
    match value {
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                return Ok(1);
            }
            let first_depth = competing_risks_array_depth(&items[0], &format!("{key}[0]"))?;
            for (idx, item) in items.iter().enumerate().skip(1) {
                let depth = competing_risks_array_depth(item, &format!("{key}[{idx}]"))?;
                if depth != first_depth {
                    return Err(py_value_error(format!(
                        "{key} has inconsistent array depth at index {idx}: got {depth}, expected {first_depth}"
                    )));
                }
            }
            Ok(first_depth + 1)
        }
        serde_json::Value::Number(_) => Ok(0),
        _ => Err(py_value_error(format!("{key} must contain only numeric arrays"))),
    }
}

fn competing_risks_number(value: &serde_json::Value, key: &str) -> PyResult<f64> {
    let number = value
        .as_f64()
        .ok_or_else(|| py_value_error(format!("{key} must be a finite number")))?;
    if !number.is_finite() {
        return Err(py_value_error(format!("{key} must be finite")));
    }
    Ok(number)
}

#[pyfunction]
fn extract_row_ids(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    id_column: Option<String>,
) -> PyResult<Option<Vec<String>>> {
    let Some(id_column) = id_column else {
        return Ok(None);
    };
    let index = headers
        .iter()
        .position(|header| header == &id_column)
        .ok_or_else(|| {
            py_value_error(format!(
                "id_column '{id_column}' is missing from prediction data"
            ))
        })?;
    let mut row_ids = Vec::with_capacity(rows.len());
    for (row_idx, row) in rows.iter().enumerate() {
        let value = row.get(index).ok_or_else(|| {
            py_value_error(format!(
                "row {row_idx} is missing id_column '{id_column}' at index {index}"
            ))
        })?;
        row_ids.push(value.clone());
    }
    Ok(Some(row_ids))
}

fn python_string_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut repr = String::with_capacity(value.len() + 2);
    repr.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => repr.push_str("\\\\"),
            '\t' => repr.push_str("\\t"),
            '\n' => repr.push_str("\\n"),
            '\r' => repr.push_str("\\r"),
            '\'' if quote == '\'' => repr.push_str("\\'"),
            '"' if quote == '"' => repr.push_str("\\\""),
            other => repr.push(other),
        }
    }
    repr.push(quote);
    repr
}

fn python_float_display(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1.0e16 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

#[pyfunction]
fn default_survival_time_grid(
    model_class: &str,
    formula: &str,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> PyResult<Option<Vec<f64>>> {
    match model_class {
        "survival"
        | "competing risks survival"
        | "survival marginal-slope"
        | "survival location-scale" => {}
        _ => return Ok(None),
    }

    let re = Regex::new(r"\s*Surv\s*\(\s*([^\s,]+)\s*,\s*([^\s,]+)\s*,\s*[^\s,]+\s*\)")
        .map_err(|err| py_value_error(format!("invalid survival formula regex: {err}")))?;
    let Some(captures) = re.captures(formula) else {
        return Ok(None);
    };
    let entry_name = captures
        .get(1)
        .map(|matched| matched.as_str())
        .unwrap_or_default();
    let exit_name = captures
        .get(2)
        .map(|matched| matched.as_str())
        .unwrap_or_default();

    let header_to_index: HashMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    let entry_idx = header_to_index.get(entry_name).copied();
    let exit_idx = header_to_index.get(exit_name).copied();
    if entry_idx.is_none() || exit_idx.is_none() {
        let mut missing = Vec::new();
        if entry_idx.is_none() {
            missing.push(entry_name);
        }
        if exit_idx.is_none() {
            missing.push(exit_name);
        }
        return Err(py_value_error(format!(
            "survival prediction data is missing required time column(s): {}",
            missing.join(", ")
        )));
    }
    let entry_idx = entry_idx.unwrap();
    let exit_idx = exit_idx.unwrap();

    let entry_name_repr = python_string_repr(entry_name);
    let exit_name_repr = python_string_repr(exit_name);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut observed = false;
    for (row_index, row) in rows.iter().enumerate() {
        let entry_cell = row.get(entry_idx).ok_or_else(|| {
            py_value_error(format!(
                "survival entry column {entry_name_repr} is missing at row {}",
                row_index + 1
            ))
        })?;
        let exit_cell = row.get(exit_idx).ok_or_else(|| {
            py_value_error(format!(
                "survival exit column {exit_name_repr} is missing at row {}",
                row_index + 1
            ))
        })?;
        let entry_value = entry_cell.parse::<f64>().map_err(|_| {
            let entry_cell_repr = python_string_repr(entry_cell);
            py_value_error(format!(
                "survival entry column {entry_name_repr} has a non-numeric value at row {}: {entry_cell_repr}",
                row_index + 1
            ))
        })?;
        let exit_value = exit_cell.parse::<f64>().map_err(|_| {
            let exit_cell_repr = python_string_repr(exit_cell);
            py_value_error(format!(
                "survival exit column {exit_name_repr} has a non-numeric value at row {}: {exit_cell_repr}",
                row_index + 1
            ))
        })?;
        if !entry_value.is_finite() || !exit_value.is_finite() {
            return Err(py_value_error(
                "survival time columns must contain only finite values".to_string(),
            ));
        }
        lo = lo.min(entry_value);
        hi = hi.max(exit_value);
        observed = true;
    }
    if !observed {
        return Ok(None);
    }
    if hi <= lo {
        let lo_display = python_float_display(lo);
        let hi_display = python_float_display(hi);
        return Err(py_value_error(format!(
            "survival exit times must extend beyond entry times; got min entry {lo_display} and max exit {hi_display}"
        )));
    }
    let span = hi - lo;
    let hi_padded = hi + (span * 1.0e-6).max(1.0e-9);
    let step = (hi_padded - lo) / 63.0;
    Ok(Some(
        (0..64).map(|index| lo + step * (index as f64)).collect(),
    ))
}

#[pyfunction(signature = (headers, rows, formula, config_json = None, fisher_rao_w = None))]
fn fit_table(
    py: Python<'_>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<String>,
    fisher_rao_w: Option<PyReadonlyArray3<'_, f64>>,
) -> PyResult<Py<PyBytes>> {
    // PyO3 0.28 names the old `allow_threads` API `detach`: the closure
    // runs without the GIL, so Python signal handling (KeyboardInterrupt,
    // SIGALRM handlers, etc.) can run while the Rust solver is in progress.
    let fisher_values = fisher_rao_w.as_ref().map(|w| w.as_array().to_owned());
    let model_bytes = detach_py_result(py, "fit_table", move || {
        fit_table_impl(
            headers,
            rows,
            formula,
            config_json.as_deref(),
            fisher_values.as_ref().map(|w| w.view()),
        )
    })?;
    Ok(PyBytes::new(py, &model_bytes).unbind())
}

#[pyfunction(signature = (x, y, formula, config_json = None, fisher_rao_w = None))]
fn fit_array(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    formula: String,
    config_json: Option<String>,
    fisher_rao_w: Option<PyReadonlyArray3<'_, f64>>,
) -> PyResult<Py<PyBytes>> {
    let x_values = x.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let fisher_values = fisher_rao_w.as_ref().map(|w| w.as_array().to_owned());
    let model_bytes = detach_py_result(py, "fit_array", move || {
        fit_array_impl(
            x_values.view(),
            y_values.view(),
            formula,
            config_json.as_deref(),
            fisher_values.as_ref().map(|w| w.view()),
        )
    })?;
    Ok(PyBytes::new(py, &model_bytes).unbind())
}

#[pyfunction]
fn load_model(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<()> {
    detach_py_result(py, "load_model", move || {
        load_model_impl(&model_bytes).map(drop)
    })
}

#[pyfunction]
fn bayes_factor_log_diff(model_a_bytes: Vec<u8>, model_b_bytes: Vec<u8>) -> PyResult<f64> {
    let model_a = load_model_impl(&model_a_bytes).map_err(PyValueError::new_err)?;
    let model_b = load_model_impl(&model_b_bytes).map_err(PyValueError::new_err)?;
    let fit_a =
        fit_result_from_saved_model_for_prediction(&model_a).map_err(PyValueError::new_err)?;
    let fit_b =
        fit_result_from_saved_model_for_prediction(&model_b).map_err(PyValueError::new_err)?;
    Ok(fit_a.reml_score - fit_b.reml_score)
}

#[pyfunction]
fn saved_model_payload_string(model_bytes: Vec<u8>, key: &str) -> PyResult<Option<String>> {
    let saved: serde_json::Value = serde_json::from_slice(&model_bytes)
        .map_err(|err| PyValueError::new_err(format!("saved model payload must be JSON: {err}")))?;
    let payload = saved
        .get("payload")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            PyValueError::new_err("saved model payload is missing its payload object")
        })?;
    Ok(payload.get(key).map(|value| match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }))
}

#[pyfunction]
fn build_extend_group_payload_json(
    spec_json: &str,
    metadata_json: Option<String>,
    prior_json: Option<String>,
) -> PyResult<String> {
    let spec_value: serde_json::Value = serde_json::from_str(spec_json)
        .map_err(|err| py_value_error(format!("invalid new_group_spec json: {err}")))?;
    let serde_json::Value::Object(mut payload) = spec_value else {
        return Err(py_value_error(
            "new_group_spec json must be an object".to_string(),
        ));
    };
    if let Some(raw) = metadata_json {
        let metadata: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|err| py_value_error(format!("invalid metadata json: {err}")))?;
        payload.insert("metadata".to_string(), metadata);
    }
    if let Some(raw) = prior_json {
        let prior: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|err| py_value_error(format!("invalid prior json: {err}")))?;
        payload.insert("prior".to_string(), prior);
    }
    serde_json::to_string(&payload)
        .map_err(|err| py_value_error(format!("failed to serialize extend group payload: {err}")))
}

#[pyfunction]
fn extend_model_with_group(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    request_json: String,
) -> PyResult<Py<PyBytes>> {
    let out = detach_py_result(py, "extend_model_with_group", move || {
        extend_model_with_group_impl(&model_bytes, &request_json)
    })?;
    Ok(PyBytes::new(py, &out).unbind())
}

#[pyfunction]
fn validate_formula_json(
    py: Python<'_>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<String>,
) -> PyResult<String> {
    detach_py_result(py, "validate_formula_json", move || {
        validate_formula_json_impl(headers, rows, formula, config_json.as_deref())
    })
}

#[pyfunction]
fn formula_validation_supported_by_python_json(payload_json: String) -> PyResult<bool> {
    let payload = parse_formula_validation_payload_json(&payload_json).map_err(py_value_error)?;
    Ok(payload
        .get("supported_by_python")
        .map(json_payload_truthy)
        .unwrap_or(false))
}

#[pyfunction]
fn formula_validation_repr_json(payload_json: String) -> PyResult<String> {
    let payload = parse_formula_validation_payload_json(&payload_json).map_err(py_value_error)?;
    let supported_by_python = payload
        .get("supported_by_python")
        .map(json_payload_truthy)
        .unwrap_or(false);
    Ok(format!(
        "FormulaValidation(formula={}, model_class={}, family_name={}, supported_by_python={})",
        python_repr_json_value(payload.get("formula")),
        python_repr_json_value(payload.get("model_class")),
        python_repr_json_value(payload.get("family_name")),
        if supported_by_python { "True" } else { "False" },
    ))
}

#[pyfunction]
fn formula_validation_html_json(payload_json: String) -> PyResult<String> {
    // Pure presentation layer; no math.
    let payload = parse_formula_validation_payload_json(&payload_json).map_err(py_value_error)?;
    let mut rows = String::new();
    for (key, value) in payload.iter() {
        rows.push_str("<tr>");
        rows.push_str(
            "<th style='text-align:left;padding:0.25rem 0.75rem 0.25rem 0;'>",
        );
        rows.push_str(&escape_html(key));
        rows.push_str("</th><td style='padding:0.25rem 0;'>");
        rows.push_str(&escape_html(&python_str_json_value(value)));
        rows.push_str("</td></tr>");
    }
    Ok(format!(
        "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>\
<h3 style='margin:0 0 0.5rem 0;'>Formula Validation</h3>\
<table style='border-collapse:collapse;'>{rows}</table></div>"
    ))
}

#[pyfunction]
fn build_predict_payload_json(
    interval: Option<f64>,
    with_uncertainty: bool,
    time_grid: Option<Vec<f64>>,
) -> PyResult<String> {
    let payload = PyPredictOptionsPayload {
        interval,
        with_uncertainty,
        time_grid,
    };
    serde_json::to_string(&payload)
        .map_err(|err| py_value_error(format!("failed to serialize predict payload: {err}")))
}

#[pyfunction]
fn predict_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<String>,
) -> PyResult<String> {
    detach_py_result(py, "predict_table", move || {
        predict_table_impl(&model_bytes, headers, rows, options_json.as_deref())
    })
}

#[pyfunction]
fn predict_array<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
    x: PyReadonlyArray2<'py, f64>,
    options_json: Option<String>,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_values = x.as_array().to_owned();
    let out = detach_py_result(py, "predict_array", move || {
        predict_array_impl(&model_bytes, x_values.view(), options_json.as_deref())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn competing_risks_cif<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    cumulative_hazards: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray2<f64>>)> {
    let time_values = times.as_array().to_owned();
    let cumulative_hazard_values = cumulative_hazards
        .iter()
        .map(|hazard| hazard.as_array().to_owned())
        .collect::<Vec<_>>();
    let (cif, overall_survival) = detach_py_result(py, "competing_risks_cif", move || {
        competing_risks_cif_impl(time_values.view(), &cumulative_hazard_values)
    })?;
    Ok((
        cif.into_pyarray(py).unbind(),
        overall_survival.into_pyarray(py).unbind(),
    ))
}

fn competing_risks_cif_impl(
    times: ArrayView1<'_, f64>,
    cumulative_hazards: &[Array2<f64>],
) -> Result<(Array3<f64>, Array2<f64>), String> {
    let endpoint_views = cumulative_hazards
        .iter()
        .map(|hazard| hazard.view())
        .collect::<Vec<_>>();
    let cumulative_hazard =
        ndarray::stack(Axis(0), &endpoint_views).map_err(|err| err.to_string())?;
    let result = gam::survival::assemble_competing_risks_cif(times, cumulative_hazard.view())
        .map_err(|err| err.to_string())?;
    Ok((result.cif, result.overall_survival))
}

#[pyfunction]
fn competing_risks_cif_from_predictions<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    cumulative_hazards: Vec<PyReadonlyArray2<'py, f64>>,
    endpoint_names: Vec<String>,
) -> PyResult<(PyObject, PyObject)> {
    if cumulative_hazards.is_empty() {
        return Err(py_value_error(
            "competing_risks_cif requires at least one endpoint prediction".to_string(),
        ));
    }
    if endpoint_names.len() != cumulative_hazards.len() {
        return Err(py_value_error(
            "endpoint_names must match the number of endpoint predictions".to_string(),
        ));
    }
    let unique_endpoint_names = endpoint_names.iter().collect::<BTreeSet<_>>();
    if unique_endpoint_names.len() != endpoint_names.len() {
        return Err(py_value_error("endpoint_names must be unique".to_string()));
    }

    let time_values = times.as_array().to_owned();
    if time_values.iter().any(|time| !time.is_finite()) {
        return Err(py_value_error(
            "time grid must contain only finite values".to_string(),
        ));
    }

    let expected_shape = cumulative_hazards[0].as_array().dim();
    let (n_rows, n_times) = expected_shape;
    if n_rows == 0 || n_times == 0 {
        return Err(py_value_error(
            "endpoint predictions must have non-empty (n_rows, n_times) shape".to_string(),
        ));
    }
    if time_values.len() != n_times {
        return Err(py_value_error(
            "time grid length must match endpoint prediction column count".to_string(),
        ));
    }
    for cumulative_hazard in cumulative_hazards.iter().skip(1) {
        if cumulative_hazard.as_array().dim() != expected_shape {
            return Err(py_value_error(
                "all endpoint predictions must return the same (n_rows, n_times) shape"
                    .to_string(),
            ));
        }
    }

    let cumulative_hazard_values = cumulative_hazards
        .iter()
        .map(|hazard| hazard.as_array().to_owned())
        .collect::<Vec<_>>();
    let (cif, overall_survival) = detach_py_result(
        py,
        "competing_risks_cif_from_predictions",
        move || {
            competing_risks_cif_from_predictions_impl(
                time_values.view(),
                &cumulative_hazard_values,
            )
        },
    )?;
    let cif_arrays = PyList::empty(py);
    for endpoint_cif in cif {
        cif_arrays.append(endpoint_cif.into_pyarray(py))?;
    }
    Ok((
        cif_arrays.unbind().into_any(),
        overall_survival.into_pyarray(py).unbind().into_any(),
    ))
}

fn competing_risks_cif_from_predictions_impl(
    times: ArrayView1<'_, f64>,
    cumulative_hazards: &[Array2<f64>],
) -> Result<(Vec<Array2<f64>>, Array2<f64>), String> {
    let endpoint_views = cumulative_hazards
        .iter()
        .map(|hazard| hazard.view())
        .collect::<Vec<_>>();
    let result =
        gam::survival::assemble_competing_risks_cif_from_endpoints(times, &endpoint_views)
            .map_err(|err| err.to_string())?;
    let cif = result
        .cif
        .axis_iter(Axis(0))
        .map(|endpoint_cif| endpoint_cif.to_owned())
        .collect::<Vec<_>>();
    Ok((cif, result.overall_survival))
}

#[pyfunction]
fn build_sample_payload_json(
    samples: Option<i64>,
    warmup: Option<i64>,
    chains: Option<i64>,
    target_accept: Option<f64>,
    seed: Option<i64>,
) -> PyResult<String> {
    let mut payload = serde_json::Map::new();
    if let Some(value) = samples {
        payload.insert("samples".to_string(), serde_json::Value::from(value));
    }
    if let Some(value) = warmup {
        payload.insert("warmup".to_string(), serde_json::Value::from(value));
    }
    if let Some(value) = chains {
        payload.insert("chains".to_string(), serde_json::Value::from(value));
    }
    if let Some(value) = target_accept {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| py_value_error("target_accept must be finite".to_string()))?;
        payload.insert(
            "target_accept".to_string(),
            serde_json::Value::Number(number),
        );
    }
    if let Some(value) = seed {
        payload.insert("seed".to_string(), serde_json::Value::from(value));
    }
    serde_json::to_string(&serde_json::Value::Object(payload)).map_err(|err| {
        py_value_error(format!(
            "failed to serialize sample options json: {err}"
        ))
    })
}

#[pyfunction]
fn sample_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<String>,
) -> PyResult<String> {
    detach_py_result(py, "sample_table", move || {
        sample_table_impl(&model_bytes, headers, rows, options_json.as_deref())
    })
}

// paired_sample_table and paired_cumulative_incidence_table pyffi wrappers
// removed: their payload types (PairedSamplePayload, PairedCifPayload,
// PyPairedCifOptions) were never defined in the main crate; the orphaned
// implementations and their helpers were deleted below.

#[pyfunction]
fn design_matrix_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> PyResult<String> {
    detach_py_result(py, "design_matrix_table", move || {
        design_matrix_table_impl(&model_bytes, headers, rows)
    })
}

#[pyfunction]
fn design_matrix_table_dense(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let out = detach_py_result(py, "design_matrix_table_dense", move || {
        let raw = design_matrix_table_impl(&model_bytes, headers, rows)?;
        design_matrix_payload_to_dense(&raw)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn design_matrix_array<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_values = x.as_array().to_owned();
    let out = detach_py_result(py, "design_matrix_array", move || {
        design_matrix_array_impl(&model_bytes, x_values.view())
    })?;
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

#[pyfunction]
fn periodic_basis_with_jet<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    n_harmonics: usize,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray3<f64>>,
    Py<PyArray2<f64>>,
)> {
    let t = t.as_array();
    if t.iter().any(|value| !value.is_finite()) {
        return Err(py_value_error(
            "periodic_basis_with_jet requires finite t values".to_string(),
        ));
    }

    let n_rows = t.len();
    let n_cols = 1 + 2 * n_harmonics;
    let mut phi = Array2::<f64>::zeros((n_rows, n_cols));
    let mut jet = Array3::<f64>::zeros((n_rows, n_cols, 1));
    let mut penalty = Array2::<f64>::zeros((n_cols, n_cols));

    phi.column_mut(0).fill(1.0);
    penalty[[0, 0]] = 1.0e-8;

    for h in 1..=n_harmonics {
        let h_f = h as f64;
        let frequency = std::f64::consts::TAU * h_f;
        let sin_col = 1 + 2 * (h - 1);
        let cos_col = sin_col + 1;
        let harmonic_penalty = h_f * h_f * h_f * h_f;

        penalty[[sin_col, sin_col]] = harmonic_penalty;
        penalty[[cos_col, cos_col]] = harmonic_penalty;

        for row in 0..n_rows {
            let angle = frequency * t[row];
            let sin_value = angle.sin();
            let cos_value = angle.cos();

            phi[[row, sin_col]] = sin_value;
            phi[[row, cos_col]] = cos_value;
            jet[[row, sin_col, 0]] = frequency * cos_value;
            jet[[row, cos_col, 0]] = -frequency * sin_value;
        }
    }

    Ok((
        phi.into_pyarray(py).unbind(),
        jet.into_pyarray(py).unbind(),
        penalty.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (points, centers, m = 2))]
fn duchon_basis_with_jet<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    m: usize,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray3<f64>>,
    Py<PyArray2<f64>>,
)> {
    if m == 0 {
        return Err(py_value_error("Duchon m must be at least 1".to_string()));
    }
    let pts = points.as_array();
    let ctrs = centers.as_array();
    if pts.ncols() != ctrs.ncols() {
        return Err(py_value_error(format!(
            "points has d={} but centers has d={}",
            pts.ncols(),
            ctrs.ncols()
        )));
    }
    if pts.iter().any(|value| !value.is_finite()) || ctrs.iter().any(|value| !value.is_finite()) {
        return Err(py_value_error(
            "duchon_basis_with_jet requires finite points and centers".to_string(),
        ));
    }

    let requested_nullspace = duchon_nullspace_from_m(m);
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
        length_scale: None,
        power: 0.0,
        nullspace_order: requested_nullspace,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    let built = build_duchon_basis(pts, &spec).map_err(|err| py_value_error(err.to_string()))?;
    let phi = built.design.to_dense();
    let primary_idx = built
        .penaltyinfo
        .iter()
        .position(|info| matches!(info.source, gam::basis::PenaltySource::Primary))
        .ok_or_else(|| {
            py_value_error("duchon_basis_with_jet: primary penalty was not built".to_string())
        })?;
    let penalty = built.penalties[primary_idx].clone();

    let effective_nullspace =
        pyffi_duchon_effective_nullspace_order(ctrs, requested_nullspace);
    let radial_transform = pyffi_duchon_kernel_constraint_nullspace(ctrs, effective_nullspace)
        .map_err(py_value_error)?;
    let radial_first = duchon_radial_first_derivative_nd(pts, ctrs, None, effective_nullspace)
        .map_err(|err| py_value_error(err.to_string()))?;
    let radial_jet = radial_input_location_jet(pts, ctrs, radial_first.view())
        .map_err(py_value_error)?;
    let poly_jet = duchon_polynomial_first_derivative_nd(pts, effective_nullspace);

    let n_rows = pts.nrows();
    let dim = pts.ncols();
    let n_kernel = radial_transform.ncols();
    let n_poly = poly_jet.shape()[1];
    let mut jet = Array3::<f64>::zeros((n_rows, n_kernel + n_poly, dim));
    for axis in 0..dim {
        let projected = radial_jet.index_axis(Axis(2), axis).dot(&radial_transform);
        jet.slice_mut(s![.., ..n_kernel, axis]).assign(&projected);
    }
    jet.slice_mut(s![.., n_kernel.., ..]).assign(&poly_jet);

    if phi.ncols() != jet.shape()[1] {
        return Err(py_value_error(format!(
            "duchon_basis_with_jet shape mismatch: Phi has {} columns but Jet has {}",
            phi.ncols(),
            jet.shape()[1]
        )));
    }
    if penalty.nrows() != phi.ncols() || penalty.ncols() != phi.ncols() {
        return Err(py_value_error(format!(
            "duchon_basis_with_jet penalty shape mismatch: expected {}x{}, got {}x{}",
            phi.ncols(),
            phi.ncols(),
            penalty.nrows(),
            penalty.ncols()
        )));
    }

    Ok((
        phi.into_pyarray(py).unbind(),
        jet.into_pyarray(py).unbind(),
        penalty.into_pyarray(py).unbind(),
    ))
}

fn required_usize_param(params: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    params
        .get_item(key)?
        .ok_or_else(|| py_value_error(format!("basis_with_jet params missing {key:?}")))?
        .extract::<usize>()
}

#[pyfunction(signature = (kind, t, params))]
fn basis_with_jet<'py>(
    py: Python<'py>,
    kind: &str,
    t: PyReadonlyArray2<'py, f64>,
    params: &Bound<'py, PyDict>,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray3<f64>>,
    Py<PyArray2<f64>>,
)> {
    match kind.to_ascii_lowercase().replace('-', "_").as_str() {
        "duchon" | "euclidean" | "euclidean_patch" => {
            let centers = params
                .get_item("centers")?
                .ok_or_else(|| py_value_error("basis_with_jet params missing \"centers\"".to_string()))?
                .extract::<PyReadonlyArray2<'py, f64>>()?;
            let m = required_usize_param(params, "m")?;
            duchon_basis_with_jet(py, t, centers, m)
        }
        "periodic" | "periodic_spline" | "circle" => {
            let n_harmonics = required_usize_param(params, "n_harmonics")?;
            let coords = t.as_array();
            if coords.ncols() == 0 {
                return Err(py_value_error(
                    "basis_with_jet periodic basis expects at least one t column".to_string(),
                ));
            }
            if coords.iter().any(|value| !value.is_finite()) {
                return Err(py_value_error(
                    "basis_with_jet periodic basis requires finite t values".to_string(),
                ));
            }

            let n_rows = coords.nrows();
            let n_cols = 1 + 2 * n_harmonics;
            let mut phi = Array2::<f64>::zeros((n_rows, n_cols));
            let mut jet = Array3::<f64>::zeros((n_rows, n_cols, 1));
            let mut penalty = Array2::<f64>::zeros((n_cols, n_cols));
            phi.column_mut(0).fill(1.0);
            penalty[[0, 0]] = 1.0e-8;

            for h in 1..=n_harmonics {
                let h_f = h as f64;
                let frequency = std::f64::consts::TAU * h_f;
                let sin_col = 1 + 2 * (h - 1);
                let cos_col = sin_col + 1;
                let harmonic_penalty = h_f * h_f * h_f * h_f;
                penalty[[sin_col, sin_col]] = harmonic_penalty;
                penalty[[cos_col, cos_col]] = harmonic_penalty;

                for row in 0..n_rows {
                    let angle = frequency * coords[[row, 0]].rem_euclid(1.0);
                    let sin_value = angle.sin();
                    let cos_value = angle.cos();
                    phi[[row, sin_col]] = sin_value;
                    phi[[row, cos_col]] = cos_value;
                    jet[[row, sin_col, 0]] = frequency * cos_value;
                    jet[[row, cos_col, 0]] = -frequency * sin_value;
                }
            }

            Ok((
                phi.into_pyarray(py).unbind(),
                jet.into_pyarray(py).unbind(),
                penalty.into_pyarray(py).unbind(),
            ))
        }
        "sphere" => sphere_chart_basis_with_jet(py, t),
        other => Err(py_value_error(format!(
            "basis_with_jet unsupported basis kind {other:?}"
        ))),
    }
}

/// Evaluate the Duchon m-spline basis at `points` against K `centers`,
/// for any input dimensionality `d ≥ 1`.
///
/// `points` is `(N, d)`, `centers` is `(K, d)`. For 1D smooths, pass
/// shapes `(N, 1)` and `(K, 1)`.
///
/// `periodic_per_axis` is an optional `Vec<bool>` of length `d`. Whenever any
/// axis is periodic (including the 1D circle case), the mixed-periodicity
/// radial polyharmonic builder is used (cylinder/torus chord distance);
/// per-axis periods are auto-derived from the centers' span along each
/// periodic axis.
#[pyfunction(signature = (
    points,
    centers,
    m = 2,
    periodic_per_axis = None,
    length_scale = None,
    nullspace_order = None,
    power = None,
))]
fn duchon_basis<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    m: usize,
    periodic_per_axis: Option<Vec<bool>>,
    length_scale: Option<f64>,
    nullspace_order: Option<&str>,
    power: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    if m == 0 {
        return Err(py_value_error("Duchon m must be at least 1".to_string()));
    }
    let pts = points.as_array();
    let ctrs = centers.as_array();
    if pts.ncols() != ctrs.ncols() {
        return Err(py_value_error(format!(
            "points has d={} but centers has d={}",
            pts.ncols(),
            ctrs.ncols()
        )));
    }
    let d = pts.ncols();
    let periodic_flags = periodic_per_axis.unwrap_or_else(|| vec![false; d]);
    if periodic_flags.len() != d {
        return Err(py_value_error(format!(
            "periodic_per_axis must have length d={}, got {}",
            d,
            periodic_flags.len()
        )));
    }
    let any_periodic = periodic_flags.iter().any(|&b| b);
    // Preserve the pre-change default path bit-for-bit when no hybrid /
    // explicit-order keyword is supplied: `length_scale=None`,
    // `power=0.0`, and `nullspace_order = duchon_nullspace_from_m(m)`.
    // When any of the three new keywords is set, route through the
    // shared hybrid resolver (matches the formula API).
    let hybrid_requested = length_scale.is_some() || nullspace_order.is_some() || power.is_some();
    let (spec_length_scale, spec_nullspace, spec_power) = if hybrid_requested {
        let cfg = resolve_duchon_hybrid_config(d, m, length_scale, nullspace_order, power)?;
        (cfg.length_scale, cfg.nullspace_order, cfg.power)
    } else {
        (None, duchon_nullspace_from_m(m), 0.0)
    };
    // Any periodic axis (1D or multi-D) routes through the mixed-periodicity
    // builder (cylinder/torus chord-distance polyharmonic).
    if any_periodic {
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
            length_scale: spec_length_scale,
            power: spec_power,
            nullspace_order: spec_nullspace,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: Default::default(),
            periodic: None,
            boundary: OneDimensionalBoundary::Open,
        };
        let built = build_duchon_basis_mixed_periodicity_auto(pts, &spec, &periodic_flags, None)
            .map_err(|err| py_value_error(err.to_string()))?;
        return Ok(built.design.to_dense().into_pyarray(py).unbind());
    }
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
        length_scale: spec_length_scale,
        power: spec_power,
        nullspace_order: spec_nullspace,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    let built = build_duchon_basis(pts, &spec).map_err(|err| py_value_error(err.to_string()))?;
    Ok(built.design.to_dense().into_pyarray(py).unbind())
}

#[pyfunction(signature = (t, num_internal_knots, degree = 3))]
fn auto_knots_1d<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    num_internal_knots: usize,
    degree: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let knots = auto_knot_vector_1d_quantile(t.as_array(), num_internal_knots, degree)
        .map_err(|err| py_value_error(err.to_string()))?;
    Ok(knots.into_pyarray(py).unbind())
}

#[pyfunction(signature = (t, num_centers))]
fn auto_centers_1d<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    num_centers: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let centers = auto_centers_1d_equal_mass(t.as_array(), num_centers)
        .map_err(|err| py_value_error(err.to_string()))?;
    Ok(centers.into_pyarray(py).unbind())
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

#[pyfunction(signature = (centers, m = 2, periodic = false, period = None))]
fn duchon_operator_penalties<'py>(
    py: Python<'py>,
    centers: PyReadonlyArray1<'py, f64>,
    m: usize,
    periodic: bool,
    period: Option<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    if periodic {
        return Err(py_value_error(
            "periodic Duchon operator penalties are not defined for the triple-operator collocation constructor"
                .to_string(),
        ));
    } else {
        validate_position_period("duchon", centers.as_array(), false, period)
            .map_err(py_value_error)?;
    }
    if m == 0 {
        return Err(py_value_error("Duchon m must be at least 1".to_string()));
    }
    let center_matrix = column_array(centers.as_array());
    let matrices = build_duchon_operator_penalty_matrices(
        center_matrix.view(),
        None,
        None,
        0.0,
        duchon_nullspace_from_m(m),
        None,
        None,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    Ok((
        matrices.mass.into_pyarray(py).unbind(),
        matrices.tension.into_pyarray(py).unbind(),
        matrices.stiffness.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (
    centers,
    m = 2,
    periodic = false,
    period = None,
    periodic_per_axis = None,
    length_scale = None,
    nullspace_order = None,
    power = None,
))]
fn duchon_function_norm_penalty<'py>(
    py: Python<'py>,
    centers: &Bound<'py, PyAny>,
    m: usize,
    periodic: bool,
    period: Option<f64>,
    periodic_per_axis: Option<Vec<bool>>,
    length_scale: Option<f64>,
    nullspace_order: Option<&str>,
    power: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    if m == 0 {
        return Err(py_value_error("Duchon m must be at least 1".to_string()));
    }
    // Accept centers as either 1D (K,) or 2D (K, d). Promote 1D to (K, 1)
    // for backward compatibility.
    let center_matrix: Array2<f64> = match centers.extract::<PyReadonlyArray2<f64>>() {
        Ok(arr2) => arr2.as_array().to_owned(),
        Err(arr2_err) => {
            let arr1 = centers
                .extract::<PyReadonlyArray1<f64>>()
                .map_err(|arr1_err| {
                    py_value_error(format!(
                        "centers must be a 1D (K,) or 2D (K, d) float array; \
                     2D extraction failed: {arr2_err}; 1D extraction failed: {arr1_err}"
                    ))
                })?;
            column_array(arr1.as_array())
        }
    };
    let d = center_matrix.ncols();
    // Resolve per-axis periodicity. `periodic_per_axis` (if given) takes
    // precedence over the legacy scalar `periodic` flag. For backward
    // compatibility, when neither is given (d=1, periodic=false) we treat
    // periodicity as false; when scalar `periodic=true` (only valid for d=1)
    // we map it to `[true]`.
    let periodic_flags: Vec<bool> = if let Some(flags) = periodic_per_axis.clone() {
        if flags.len() != d {
            return Err(py_value_error(format!(
                "periodic_per_axis must have length d={}, got {}",
                d,
                flags.len()
            )));
        }
        flags
    } else if periodic {
        if d != 1 {
            return Err(py_value_error(
                "scalar `periodic=true` only valid for 1D centers; \
                 use `periodic_per_axis` for d > 1"
                    .to_string(),
            ));
        }
        vec![true]
    } else {
        vec![false; d]
    };
    let any_periodic = periodic_flags.iter().any(|&b| b);
    // Validate period only for the 1D periodic case (matches the legacy
    // 1D validator's contract). For d > 1 mixed-periodicity, per-axis
    // periods are auto-derived from the centers in the Rust core; the
    // scalar `period` is only meaningful in the 1D legacy path.
    if d == 1 {
        let col = center_matrix.column(0);
        validate_position_period("duchon", col, any_periodic, period).map_err(py_value_error)?;
    } else if period.is_some() {
        return Err(py_value_error(
            "duchon scalar `period` is only valid for d=1 (multi-D periodic axes auto-derive period from centers)".to_string(),
        ));
    }
    // Preserve the pre-change default path bit-for-bit when no hybrid /
    // explicit-order keyword is supplied. When any of the three new
    // keywords is set, route through the shared hybrid resolver.
    let hybrid_requested = length_scale.is_some() || nullspace_order.is_some() || power.is_some();
    let (spec_length_scale, spec_nullspace, spec_power) = if hybrid_requested {
        let cfg = resolve_duchon_hybrid_config(d, m, length_scale, nullspace_order, power)?;
        (cfg.length_scale, cfg.nullspace_order, cfg.power)
    } else {
        (None, duchon_nullspace_from_m(m), 0.0)
    };
    // Any periodic axis (1D or multi-D) routes through the mixed-periodicity
    // builder (cylinder/torus chord-distance polyharmonic).
    if any_periodic {
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(center_matrix.clone()),
            length_scale: spec_length_scale,
            power: spec_power,
            nullspace_order: spec_nullspace,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: Default::default(),
            periodic: None,
            boundary: OneDimensionalBoundary::Open,
        };
        let built = build_duchon_basis_mixed_periodicity_auto(
            center_matrix.view(),
            &spec,
            &periodic_flags,
            None,
        )
        .map_err(|err| py_value_error(err.to_string()))?;
        // Mixed-periodicity builder emits a single Primary candidate (the
        // function-norm Gram).
        let idx = built
            .penaltyinfo
            .iter()
            .position(|info| matches!(info.source, gam::basis::PenaltySource::Primary))
            .ok_or_else(|| {
                py_value_error(
                    "mixed-periodicity Duchon function-norm penalty was not built".to_string(),
                )
            })?;
        let penalty = built.penalties[idx].clone();
        return Ok(penalty.into_pyarray(py).unbind());
    }
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(center_matrix.clone()),
        length_scale: spec_length_scale,
        power: spec_power,
        nullspace_order: spec_nullspace,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    let built = build_duchon_basis(center_matrix.view(), &spec)
        .map_err(|err| py_value_error(err.to_string()))?;
    // The scale-free pure-Duchon path emits the operator triplet (mass,
    // tension, stiffness); the function-norm seminorm is the curvature
    // (stiffness) block — `∫|∇^(p+s) f|²`-flavoured.
    let stiffness_idx = built
        .penaltyinfo
        .iter()
        .position(|info| matches!(info.source, gam::basis::PenaltySource::OperatorStiffness))
        .ok_or_else(|| {
            py_value_error(
                "Duchon function-norm penalty (stiffness) was not built; \
                 ensure spec.operator_penalties.stiffness is Active"
                    .to_string(),
            )
        })?;
    let penalty = built.penalties[stiffness_idx].clone();
    Ok(penalty.into_pyarray(py).unbind())
}

/// Build the spherical-spline (S²) basis and matching penalty matrix.
///
/// `points` is an `(N, 2)` array of latitude/longitude pairs (degrees by
/// default, radians when `radians=True`). `n_centers` controls the number
/// of Wahba centers (kernel = "sobolev" | "pseudo") or the truncation
/// degree `L` for kernel = "harmonic" (basis dim = `L * (L + 2)`).
///
/// Returns `(design, penalty)` as numpy arrays, with shapes `(N, K)` and
/// `(K, K)` respectively, where `K` is the chosen basis dimension after
/// any sum-to-zero identifiability transform applied by the Rust builder.
#[pyfunction(signature = (points, n_centers, penalty_order = 2, kernel = "sobolev", radians = false))]
fn sphere_basis<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    n_centers: usize,
    penalty_order: usize,
    kernel: &str,
    radians: bool,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let pts = points.as_array();
    if pts.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_basis expects points of shape (N, 2) [lat, lon]; got d={}",
            pts.ncols()
        )));
    }
    if !(1..=4).contains(&penalty_order) {
        return Err(py_value_error(format!(
            "sphere_basis penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        )));
    }
    let (method, wahba_kernel, max_degree) = match kernel.to_ascii_lowercase().as_str() {
        "sobolev" => (SphereMethod::Wahba, SphereWahbaKernel::Sobolev, None),
        "pseudo" => (SphereMethod::Wahba, SphereWahbaKernel::Pseudo, None),
        "harmonic" => (
            SphereMethod::Harmonic,
            SphereWahbaKernel::Sobolev,
            Some(n_centers),
        ),
        other => {
            return Err(py_value_error(format!(
                "sphere_basis kernel must be one of 'sobolev', 'pseudo', 'harmonic'; got '{other}'"
            )));
        }
    };
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: n_centers,
        },
        penalty_order,
        double_penalty: false,
        radians,
        method,
        max_degree,
        wahba_kernel,
    };
    let built =
        build_spherical_spline_basis(pts, &spec).map_err(|err| py_value_error(err.to_string()))?;
    let primary_idx = built
        .penaltyinfo
        .iter()
        .position(|info| matches!(info.source, gam::basis::PenaltySource::Primary))
        .ok_or_else(|| {
            py_value_error("sphere_basis: primary penalty was not built; check spec".to_string())
        })?;
    let penalty = built.penalties[primary_idx].clone();
    let design = built.design.to_dense();
    Ok((
        design.into_pyarray(py).unbind(),
        penalty.into_pyarray(py).unbind(),
    ))
}

/// Chart-local seven-column sphere basis with analytic lat/lon jet.
///
/// `t` is an `(N, 2)` array of latitude/longitude pairs in radians. The
/// columns are `[1, x, y, z, xy, yz, xz]` for the unit-sphere embedding.
#[pyfunction(signature = (t))]
fn sphere_chart_basis_with_jet<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>, Py<PyArray2<f64>>)> {
    let coords = t.as_array();
    if coords.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_chart_basis_with_jet expects t of shape (N, 2) [lat, lon]; got d={}",
            coords.ncols()
        )));
    }

    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 7));
    let mut jet = Array3::<f64>::zeros((n, 7, 2));
    for row in 0..n {
        let lat = coords[[row, 0]].clamp(
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
        );
        let lon = coords[[row, 1]];
        let clat = lat.cos();
        let slat = lat.sin();
        let clon = lon.cos();
        let slon = lon.sin();

        let x = clat * clon;
        let y = clat * slon;
        let z = slat;
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = x;
        phi[[row, 2]] = y;
        phi[[row, 3]] = z;
        phi[[row, 4]] = x * y;
        phi[[row, 5]] = y * z;
        phi[[row, 6]] = x * z;

        let dx_dlat = -slat * clon;
        let dx_dlon = -clat * slon;
        let dy_dlat = -slat * slon;
        let dy_dlon = clat * clon;
        let dz_dlat = clat;

        jet[[row, 1, 0]] = dx_dlat;
        jet[[row, 1, 1]] = dx_dlon;
        jet[[row, 2, 0]] = dy_dlat;
        jet[[row, 2, 1]] = dy_dlon;
        jet[[row, 3, 0]] = dz_dlat;
        jet[[row, 4, 0]] = dx_dlat * y + x * dy_dlat;
        jet[[row, 4, 1]] = dx_dlon * y + x * dy_dlon;
        jet[[row, 5, 0]] = dy_dlat * z + y * dz_dlat;
        jet[[row, 5, 1]] = dy_dlon * z;
        jet[[row, 6, 0]] = dx_dlat * z + x * dz_dlat;
        jet[[row, 6, 1]] = dx_dlon * z;
    }

    let penalty = Array2::from_diag(&Array1::from_vec(vec![
        1e-8, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0,
    ]));
    Ok((
        phi.into_pyarray(py).unbind(),
        jet.into_pyarray(py).unbind(),
        penalty.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (centers, m = 2, length_scale = 1.0))]
fn thin_plate_penalty<'py>(
    py: Python<'py>,
    centers: PyReadonlyArray2<'py, f64>,
    m: usize,
    length_scale: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    if m != 2 {
        return Err(py_value_error(
            "thin_plate_penalty currently supports only the canonical m=2 penalty".to_string(),
        ));
    }
    let matrix = build_thin_plate_penalty_matrix(centers.as_array(), length_scale)
        .map_err(|err| py_value_error(err.to_string()))?;
    Ok(matrix.penalty.into_pyarray(py).unbind())
}

#[pyfunction]
fn _block_diag<'py>(
    py: Python<'py>,
    blocks: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let mut total = 0_usize;
    for (idx, block) in blocks.iter().enumerate() {
        let view = block.as_array();
        if view.nrows() != view.ncols() {
            return Err(py_value_error(format!(
                "_block_diag block {idx} must be square; got shape ({}, {})",
                view.nrows(),
                view.ncols()
            )));
        }
        total += view.nrows();
    }

    let mut out = Array2::<f64>::zeros((total, total));
    let mut cursor = 0_usize;
    for block in blocks {
        let view = block.as_array();
        let width = view.nrows();
        out.slice_mut(s![cursor..cursor + width, cursor..cursor + width])
            .assign(&view);
        cursor += width;
    }
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (x, y, coefficients, log_lambda, penalty, weights = None, by = None, by_start_col = 0))]
fn gaussian_reml_score<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    coefficients: PyReadonlyArray2<'py, f64>,
    log_lambda: f64,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let x_values = x.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let coefficient_values = coefficients.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let score = detach_py_result(py, "gaussian_reml_score", move || {
        let gated_x;
        let fit_x = if let Some(by_arr) = by_values.as_ref() {
            gated_x = apply_by_gate(x_values.view(), by_arr.view(), by_start_col)?;
            gated_x.view()
        } else {
            x_values.view()
        };
        gaussian_reml_free_b_score(
            fit_x,
            y_values.view(),
            coefficient_values.view(),
            log_lambda,
            penalty_values.view(),
            weight_values.as_ref().map(|w| w.view()),
        )
        .map_err(|err| err.to_string())
    })?;
    let out = PyDict::new(py);
    out.set_item("reml_score", score.reml_score)?;
    out.set_item(
        "grad_coefficients",
        score.grad_coefficients.into_pyarray(py),
    )?;
    out.set_item("grad_penalty", score.grad_penalty.into_pyarray(py))?;
    out.set_item("grad_log_lambda", score.grad_log_lambda)?;
    out.set_item("fitted", score.fitted.into_pyarray(py))?;
    out.set_item("sigma2", score.sigma2.into_pyarray(py))?;
    out.set_item("edf", score.edf)?;
    Ok(out.unbind())
}

fn require_finite_matrix(name: &str, matrix: &ArrayView2<'_, f64>) -> PyResult<()> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(py_value_error(format!("{name} must contain only finite values")));
    }
    Ok(())
}

#[pyfunction(signature = (y_out, y_hat, z, w_dec, lambda_sparse, skip_u = None, skip_v = None))]
fn skip_transcoder_reml_metrics<'py>(
    py: Python<'py>,
    y_out: PyReadonlyArray2<'py, f64>,
    y_hat: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray2<'py, f64>,
    w_dec: PyReadonlyArray2<'py, f64>,
    lambda_sparse: f64,
    skip_u: Option<PyReadonlyArray2<'py, f64>>,
    skip_v: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    if !(lambda_sparse.is_finite() && lambda_sparse > 0.0) {
        return Err(py_value_error(format!(
            "lambda_sparse must be finite and > 0, got {lambda_sparse}"
        )));
    }

    let y_out = y_out.as_array();
    let y_hat = y_hat.as_array();
    let z = z.as_array();
    let w_dec = w_dec.as_array();
    let skip_u_view = skip_u.as_ref().map(|value| value.as_array());
    let skip_v_view = skip_v.as_ref().map(|value| value.as_array());

    require_finite_matrix("y_out", &y_out)?;
    require_finite_matrix("y_hat", &y_hat)?;
    require_finite_matrix("z", &z)?;
    require_finite_matrix("w_dec", &w_dec)?;
    if let Some(skip_u) = skip_u_view.as_ref() {
        require_finite_matrix("skip_u", skip_u)?;
    }
    if let Some(skip_v) = skip_v_view.as_ref() {
        require_finite_matrix("skip_v", skip_v)?;
    }

    let (n_rows, out_dim) = y_out.dim();
    if n_rows == 0 || out_dim == 0 {
        return Err(py_value_error(
            "skip_transcoder_reml_metrics requires non-empty y_out".to_string(),
        ));
    }
    if y_hat.dim() != (n_rows, out_dim) {
        return Err(py_value_error(format!(
            "y_hat shape mismatch: expected ({n_rows}, {out_dim}), got ({}, {})",
            y_hat.nrows(),
            y_hat.ncols()
        )));
    }
    if z.nrows() != n_rows {
        return Err(py_value_error(format!(
            "z row mismatch: expected {n_rows}, got {}",
            z.nrows()
        )));
    }
    if w_dec.dim() != (z.ncols(), out_dim) {
        return Err(py_value_error(format!(
            "w_dec shape mismatch: expected ({}, {out_dim}), got ({}, {})",
            z.ncols(),
            w_dec.nrows(),
            w_dec.ncols()
        )));
    }
    if let Some(skip_u) = skip_u_view.as_ref() {
        if skip_u.nrows() != out_dim {
            return Err(py_value_error(format!(
                "skip_u row mismatch: expected {out_dim}, got {}",
                skip_u.nrows()
            )));
        }
        if let Some(skip_v) = skip_v_view.as_ref() {
            if skip_v.ncols() != skip_u.ncols() {
                return Err(py_value_error(format!(
                    "skip_v rank mismatch: expected {} columns, got {}",
                    skip_u.ncols(),
                    skip_v.ncols()
                )));
            }
        }
    } else if skip_v_view.is_some() {
        return Err(py_value_error(
            "skip_v was provided without skip_u".to_string(),
        ));
    }

    let metrics = skip_transcoder_reml_metrics_core(SkipTranscoderRemlInputs {
        y_out,
        y_hat,
        z,
        w_dec,
        lambda_sparse,
        skip_u: skip_u_view,
    })
    .map_err(py_value_error)?;

    let out = PyDict::new(py);
    out.set_item("reml_score", metrics.reml_score)?;
    out.set_item("mse", metrics.mse)?;
    out.set_item("sparsity", metrics.sparsity)?;
    out.set_item("explained_variance", metrics.explained_variance)?;
    out.set_item("active_atoms", metrics.active_atoms)?;
    out.set_item("effective_rank", metrics.effective_rank)?;
    Ok(out.unbind())
}

#[pyfunction]
fn skip_transcoder_select_reml(scores: Vec<f64>) -> PyResult<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (idx, score) in scores.into_iter().enumerate() {
        if score.is_nan() {
            continue;
        }
        if best.map_or(true, |(_, best_score)| score < best_score) {
            best = Some((idx, score));
        }
    }
    best.map(|(idx, _)| idx).ok_or_else(|| {
        py_value_error("No scored candidates; call reml_score_skip_transcoder first.".to_string())
    })
}

#[pyfunction]
fn tierney_kadane_normalized_score(
    raw_reml: f64,
    null_dim: f64,
    null_space_logdet: Option<f64>,
) -> PyResult<f64> {
    gam::solver::topology_selector::tk_normalized_score(
        raw_reml,
        null_dim,
        null_space_logdet,
        1.0,
        1,
        gam::solver::evidence::TopologyScoreScale::PerObservation,
    )
    .map_err(PyValueError::new_err)
}

/// String dispatch for the torch fit entry — translate a Python `Smooth`
/// subclass name into the matching torch entry kind string.
#[pyfunction]
fn torch_smooth_dispatch_key(spec_kind: &str) -> PyResult<String> {
    gam::terms::torch_dispatch::dispatch_key(spec_kind)
        .map(|entry| entry.as_str().to_string())
        .map_err(PyValueError::new_err)
}

/// Replace the unique `s(..., type=AUTO)` term in `base_formula` with the
/// candidate-specific smooth term described by `candidate_json`. The JSON
/// payload is a typed `CandidateTopology` (tag = "kind").
///
/// Returns `Ok(Some(formula))` when the substitution succeeds, `Ok(None)`
/// when the candidate's required dimension does not match the AUTO term and
/// `strict_dimension` is false, and `Err(...)` on any other failure (missing
/// AUTO term, dimension mismatch in strict mode, malformed JSON, etc.).
#[pyfunction(signature = (base_formula, candidate_json, strict_dimension = true))]
fn assemble_candidate_formula(
    base_formula: &str,
    candidate_json: &str,
    strict_dimension: bool,
) -> PyResult<Option<String>> {
    let candidate: gam::solver::topology_formula::CandidateTopology =
        serde_json::from_str(candidate_json).map_err(|err| {
            py_value_error(format!(
                "assemble_candidate_formula: failed to parse candidate JSON: {err}"
            ))
        })?;
    gam::solver::topology_formula::assemble_candidate_formula(
        base_formula,
        &candidate,
        strict_dimension,
    )
    .map_err(PyValueError::new_err)
}

const PREFERRED_PREDICTION_COLUMNS: &[&str] =
    &["eta", "mean", "effective_se", "mean_lower", "mean_upper"];

struct OrderedPredictionColumnEntries(Vec<(String, serde_json::Value)>);

impl<'de> Deserialize<'de> for OrderedPredictionColumnEntries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OrderedPredictionColumnVisitor;

        impl<'de> Visitor<'de> for OrderedPredictionColumnVisitor {
            type Value = OrderedPredictionColumnEntries;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object containing prediction columns")
            }

            fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut entries = Vec::with_capacity(access.size_hint().unwrap_or(0));
                while let Some((key, value)) = access.next_entry::<String, serde_json::Value>()? {
                    entries.push((key, value));
                }
                Ok(OrderedPredictionColumnEntries(entries))
            }
        }

        deserializer.deserialize_map(OrderedPredictionColumnVisitor)
    }
}

fn ordered_json_object_string(
    entries: Vec<(String, serde_json::Value)>,
) -> Result<String, serde_json::Error> {
    let mut output = String::from("{");
    for (index, (key, value)) in entries.into_iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&serde_json::to_string(&key)?);
        output.push(':');
        output.push_str(&serde_json::to_string(&value)?);
    }
    output.push('}');
    Ok(output)
}

#[pyfunction]
fn ordered_prediction_columns(columns_json: &str) -> PyResult<String> {
    let OrderedPredictionColumnEntries(mut pending): OrderedPredictionColumnEntries =
        serde_json::from_str(columns_json).map_err(|err| {
            py_value_error(format!(
                "ordered_prediction_columns: failed to parse columns JSON: {err}"
            ))
        })?;
    let mut ordered = Vec::with_capacity(pending.len());
    for preferred in PREFERRED_PREDICTION_COLUMNS {
        if let Some(index) = pending
            .iter()
            .position(|entry| entry.0.as_str() == *preferred)
        {
            ordered.push(pending.remove(index));
        }
    }
    ordered.extend(pending);
    ordered_json_object_string(ordered).map_err(|err| {
        py_value_error(format!(
            "ordered_prediction_columns: failed to serialise columns JSON: {err}"
        ))
    })
}

/// Rank a set of topology candidates by their TK-normalized REML score.
///
/// Input JSON shape:
/// ```json
/// {"score_scale": "per_effective_dim" | "per_observation",
///  "candidates": [
///     {"name": "...", "raw_reml": ..., "null_dim": ...,
///      "null_space_logdet": ..., "effective_dim": ..., "n_obs": ...},
///     ...
///  ]}
/// ```
/// Output JSON: `{"ranked": [...], "winner_index": 0}` with entries sorted
/// descending by `tk_score`.
#[pyfunction]
fn rank_topology_candidates(evidence_json: &str) -> PyResult<String> {
    #[derive(Deserialize)]
    struct CandidateEvidence {
        name: String,
        raw_reml: f64,
        null_dim: f64,
        null_space_logdet: Option<f64>,
        effective_dim: f64,
        n_obs: usize,
    }
    #[derive(Deserialize)]
    struct EvidenceBundle {
        score_scale: String,
        candidates: Vec<CandidateEvidence>,
    }
    let bundle: EvidenceBundle = serde_json::from_str(evidence_json).map_err(|err| {
        py_value_error(format!(
            "rank_topology_candidates: failed to parse evidence JSON: {err}"
        ))
    })?;
    let score_scale = match bundle
        .score_scale
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "per_observation" => gam::solver::evidence::TopologyScoreScale::PerObservation,
        "per_effective_dim" => gam::solver::evidence::TopologyScoreScale::PerEffectiveDim,
        other => {
            return Err(py_value_error(format!(
                "rank_topology_candidates: score_scale must be per_effective_dim or per_observation; got {other:?}"
            )));
        }
    };
    if bundle.candidates.is_empty() {
        return Err(py_value_error(
            "rank_topology_candidates: at least one candidate is required".to_string(),
        ));
    }
    let mut ranked: Vec<serde_json::Value> = Vec::with_capacity(bundle.candidates.len());
    for entry in &bundle.candidates {
        let tk = gam::solver::topology_selector::tk_normalized_score(
            entry.raw_reml,
            entry.null_dim,
            entry.null_space_logdet,
            entry.effective_dim,
            entry.n_obs,
            score_scale,
        )
        .map_err(PyValueError::new_err)?;
        ranked.push(serde_json::json!({
            "name": entry.name,
            "tk_score": tk,
            "raw_reml": entry.raw_reml,
            "effective_dim": entry.effective_dim,
            "n_obs": entry.n_obs,
        }));
    }
    ranked.sort_by(|lhs, rhs| {
        let l = lhs.get("tk_score").and_then(|v| v.as_f64()).unwrap_or(f64::NEG_INFINITY);
        let r = rhs.get("tk_score").and_then(|v| v.as_f64()).unwrap_or(f64::NEG_INFINITY);
        r.partial_cmp(&l).unwrap_or(std::cmp::Ordering::Equal)
    });
    let out = serde_json::json!({
        "ranked": ranked,
        "winner_index": 0_usize,
    });
    serde_json::to_string(&out)
        .map_err(|err| py_value_error(format!("rank_topology_candidates: serialise: {err}")))
}

const REML_SCORE_KEYS: &[&str] = &["reml_score", "evidence", "laml", "score"];
const EDF_KEYS: &[&str] = &["edf_total", "edf", "effective_dof"];
const PENALTY_RANK_KEYS: &[&str] = &["penalty_rank", "rank_s", "rank_S", "cache_penalty_rank"];
const NULL_DIM_KEYS: &[&str] = &["null_dim"];
const NULLITY_KEYS: &[&str] = &["nullity", "penalty_nullity", "cache_nullity"];
const NULL_HESSIAN_LOGDET_KEYS: &[&str] = &[
    "null_space_logdet",
    "null_hessian_logdet",
    "h_null_logdet",
    "logdet_h_null",
];
const DIM_KEYS: &[&str] = &["effective_dim", "dim_h", "dim_H", "hessian_dim"];

struct RemlCandidateScore {
    index: usize,
    name: String,
    score: f64,
    edf: Option<f64>,
}

enum RemlFitView<'py> {
    SavedSummary(serde_json::Value),
    PythonObject(Bound<'py, PyAny>),
}

#[pyfunction]
fn extract_reml_score(py: Python<'_>, fit: Py<PyAny>) -> PyResult<f64> {
    let fit = fit.bind(py);
    extract_reml_score_impl(py, fit)
}

#[pyfunction]
fn extract_reml_score_raw(py: Python<'_>, fit: Py<PyAny>) -> PyResult<f64> {
    let fit = fit.bind(py);
    extract_reml_score_raw_impl(py, fit)
}

#[pyfunction]
fn extract_reml_edf(py: Python<'_>, fit: Py<PyAny>) -> PyResult<Option<f64>> {
    let fit = fit.bind(py);
    let view = reml_fit_view(py, fit)?;
    extract_edf_from_view(py, &view)
}

#[pyfunction(signature = (fits, names = None, cv_scores = None))]
fn compare_reml_fits(
    py: Python<'_>,
    fits: Vec<Py<PyAny>>,
    names: Option<Vec<String>>,
    cv_scores: Option<Vec<f64>>,
) -> PyResult<Py<PyDict>> {
    if fits.is_empty() {
        return Err(PyValueError::new_err("compare_models requires at least one fit"));
    }
    let labels = match names {
        Some(names) => {
            if names.len() != fits.len() {
                return Err(PyValueError::new_err(format!(
                    "len(names)={} does not match len(fits)={}",
                    names.len(),
                    fits.len()
                )));
            }
            names
        }
        None => (0..fits.len()).map(|idx| format!("fit_{idx}")).collect(),
    };
    if let Some(scores) = cv_scores.as_ref() {
        if scores.len() != fits.len() {
            return Err(PyValueError::new_err(format!(
                "len(cv_scores)={} does not match len(fits)={}",
                scores.len(),
                fits.len()
            )));
        }
    }

    let mut scored = Vec::with_capacity(fits.len());
    for (index, (name, fit)) in labels.into_iter().zip(fits.iter()).enumerate() {
        let fit = fit.bind(py);
        let view = reml_fit_view(py, fit)?;
        scored.push(RemlCandidateScore {
            index,
            name,
            score: extract_reml_score_from_view(py, &view)?,
            edf: extract_edf_from_view(py, &view)?,
        });
    }
    scored.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });

    let best_score = scored[0].score;
    let winner = scored[0].name.clone();
    let ranking = PyList::empty(py);
    let score_table = PyList::empty(py);
    for row in scored.iter() {
        let delta = row.score - best_score;
        let bayes_factor = delta.exp();
        ranking.append((
            row.name.as_str(),
            row.score,
            delta,
            bayes_factor,
            row.edf,
        ))?;

        let table_row = PyDict::new(py);
        table_row.set_item("name", row.name.as_str())?;
        table_row.set_item("reml_score", row.score)?;
        table_row.set_item("delta_reml", delta)?;
        table_row.set_item("bayes_factor_vs_best", bayes_factor)?;
        table_row.set_item("effective_dof", row.edf)?;
        score_table.append(table_row)?;
    }

    let summary = if scored.len() >= 2 {
        let runner_up = &scored[1];
        let log_bf = best_score - runner_up.score;
        format!(
            "{} wins by Bayes factor {} over {}",
            winner,
            format_bayes_factor(log_bf),
            runner_up.name
        )
    } else {
        format!("{winner} (single fit; no comparison)")
    };

    let out = PyDict::new(py);
    out.set_item("ranking", ranking)?;
    out.set_item("winner", winner)?;
    out.set_item("evidence_summary", summary)?;
    out.set_item("score_table", score_table)?;
    if let Some(scores) = cv_scores {
        let cv_optional = PyList::empty(py);
        for row in scored.iter() {
            cv_optional.append((row.name.as_str(), scores[row.index]))?;
        }
        out.set_item("cv_optional", cv_optional)?;
    }
    Ok(out.unbind())
}

fn extract_reml_score_impl(py: Python<'_>, fit: &Bound<'_, PyAny>) -> PyResult<f64> {
    let view = reml_fit_view(py, fit)?;
    extract_reml_score_from_view(py, &view)
}

fn extract_reml_score_from_view(py: Python<'_>, view: &RemlFitView<'_>) -> PyResult<f64> {
    let raw = extract_reml_score_raw_from_view(view)?;
    with_tierney_kadane_normalizer_from_view(py, view, raw)
}

fn extract_reml_score_raw_impl(py: Python<'_>, fit: &Bound<'_, PyAny>) -> PyResult<f64> {
    let view = reml_fit_view(py, fit)?;
    extract_reml_score_raw_from_view(&view)
}

fn extract_reml_score_raw_from_view(view: &RemlFitView<'_>) -> PyResult<f64> {
    if let Some(score) = extract_float_metadata_from_view(view, REML_SCORE_KEYS)? {
        return Ok(score);
    }
    match view {
        RemlFitView::SavedSummary(_) => Err(PyValueError::new_err(
            "Model summary is missing a reml_score / evidence field",
        )),
        RemlFitView::PythonObject(fit) => Err(PyTypeError::new_err(format!(
            "compare_models: cannot extract reml_score from {}; pass a gamfit.Model, \
             a dict with 'reml_score', or an object exposing .evidence",
            fit.get_type().name()?
        ))),
    }
}

fn with_tierney_kadane_normalizer_from_view(
    py: Python<'_>,
    view: &RemlFitView<'_>,
    score: f64,
) -> PyResult<f64> {
    let Some(null_dim) = extract_null_dim_from_view(py, view)? else {
        return Ok(score);
    };
    tierney_kadane_normalized_score(
        score,
        null_dim,
        extract_float_metadata_from_view(view, NULL_HESSIAN_LOGDET_KEYS)?,
    )
}

fn extract_null_dim_from_view(py: Python<'_>, view: &RemlFitView<'_>) -> PyResult<Option<f64>> {
    if let Some(null_dim) = extract_float_metadata_from_view(view, NULL_DIM_KEYS)? {
        return Ok(Some(null_dim));
    }
    if let Some(nullity) = extract_float_metadata_from_view(view, NULLITY_KEYS)? {
        return Ok(Some(nullity * extract_output_dim_from_view(py, view)?));
    }
    let dim_h = extract_float_metadata_from_view(view, DIM_KEYS)?;
    let penalty_rank = extract_float_metadata_from_view(view, PENALTY_RANK_KEYS)?;
    Ok(match (dim_h, penalty_rank) {
        (Some(dim_h), Some(penalty_rank)) => Some(dim_h - penalty_rank),
        _ => None,
    })
}

fn extract_output_dim_from_view(_py: Python<'_>, view: &RemlFitView<'_>) -> PyResult<f64> {
    match view {
        RemlFitView::SavedSummary(payload) => Ok(json_output_dim(payload)),
        RemlFitView::PythonObject(_) => {
            let Some(coefficients) = extract_py_metadata_value(view, &["coefficients"])? else {
                return Ok(1.0);
            };
            let Ok(shape) = coefficients.getattr("shape") else {
                return Ok(1.0);
            };
            let dims: Vec<usize> = shape.extract()?;
            if dims.len() >= 2 {
                Ok(dims[1] as f64)
            } else {
                Ok(1.0)
            }
        }
    }
}

fn extract_edf_from_view(_py: Python<'_>, view: &RemlFitView<'_>) -> PyResult<Option<f64>> {
    match view {
        RemlFitView::SavedSummary(payload) => Ok(json_lookup_edf(payload, EDF_KEYS)),
        RemlFitView::PythonObject(_) => {
            let Some(value) = extract_py_metadata_value(view, EDF_KEYS)? else {
                return Ok(None);
            };
            py_value_to_float_or_sum(&value).map(Some)
        }
    }
}

fn extract_float_metadata_from_view(
    view: &RemlFitView<'_>,
    keys: &[&str],
) -> PyResult<Option<f64>> {
    match view {
        RemlFitView::SavedSummary(payload) => Ok(json_lookup_f64(payload, keys)),
        RemlFitView::PythonObject(_) => {
            let Some(value) = extract_py_metadata_value(view, keys)? else {
                return Ok(None);
            };
            value.extract::<f64>().map(Some)
        }
    }
}

fn reml_fit_view<'py>(
    py: Python<'py>,
    fit: &Bound<'py, PyAny>,
) -> PyResult<RemlFitView<'py>> {
    if let Ok(model_bytes) = fit.extract::<Vec<u8>>() {
        return Ok(RemlFitView::SavedSummary(summary_payload_from_model_bytes(
            &model_bytes,
        )?));
    }
    if fit.hasattr("_model_bytes")? {
        let model_bytes: Vec<u8> = fit.getattr("_model_bytes")?.extract()?;
        return Ok(RemlFitView::SavedSummary(summary_payload_from_model_bytes(
            &model_bytes,
        )?));
    }
    if let Some(payload) = py_summary_payload(py, fit)? {
        return Ok(RemlFitView::PythonObject(payload));
    }
    Ok(RemlFitView::PythonObject(fit.clone()))
}

fn summary_payload_from_model_bytes(model_bytes: &[u8]) -> PyResult<serde_json::Value> {
    summary_payload_value_from_model_bytes(model_bytes).map_err(PyValueError::new_err)
}

fn py_summary_payload<'py>(
    _py: Python<'py>,
    fit: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !fit.hasattr("summary")? {
        return Ok(None);
    }
    let summary_method = fit.getattr("summary")?;
    if !summary_method.is_callable() {
        return Ok(None);
    }
    let summary = fit.call_method0("summary")?;
    if summary.hasattr("payload")? {
        return summary.getattr("payload").map(Some);
    }
    Ok(Some(summary))
}

fn extract_py_metadata_value<'py>(
    view: &RemlFitView<'py>,
    keys: &[&str],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let RemlFitView::PythonObject(fit) = view else {
        return Ok(None);
    };
    if let Some(value) = extract_py_get_value(fit, keys)? {
        return Ok(Some(value));
    }
    for key in keys {
        if fit.hasattr(*key)? {
            let value = fit.getattr(*key)?;
            if !value.is_none() {
                return Ok(Some(value));
            }
        }
    }
    Ok(None)
}

fn extract_py_get_value<'py>(
    target: &Bound<'py, PyAny>,
    keys: &[&str],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !target.hasattr("get")? {
        return Ok(None);
    }
    let get_method = target.getattr("get")?;
    if !get_method.is_callable() {
        return Ok(None);
    }
    for key in keys {
        let value = target.call_method1("get", (*key,))?;
        if !value.is_none() {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn py_value_to_float_or_sum(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(scalar);
    }
    let mut total = 0.0;
    for item in value.try_iter()? {
        total += item?.extract::<f64>()?;
    }
    Ok(total)
}

fn json_lookup_f64(payload: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    let object = payload.as_object()?;
    for key in keys {
        if let Some(value) = object.get(*key) {
            if let Some(value) = json_number_to_f64(value) {
                return Some(value);
            }
        }
    }
    None
}

fn json_lookup_edf(payload: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    let object = payload.as_object()?;
    for key in keys {
        if let Some(value) = object.get(*key) {
            if let Some(scalar) = json_number_to_f64(value) {
                return Some(scalar);
            }
            if let Some(values) = value.as_array() {
                return values.iter().try_fold(0.0, |acc, value| {
                    json_number_to_f64(value).map(|number| acc + number)
                });
            }
        }
    }
    None
}

fn json_number_to_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|value| value as f64))
        .or_else(|| value.as_u64().map(|value| value as f64))
}

fn json_output_dim(payload: &serde_json::Value) -> f64 {
    let Some(coefficients) = payload.get("coefficients") else {
        return 1.0;
    };
    let Some(rows) = coefficients.as_array() else {
        return 1.0;
    };
    let Some(first) = rows.first() else {
        return 1.0;
    };
    first.as_array().map_or(1.0, |row| row.len() as f64)
}

fn format_bayes_factor(log_bf: f64) -> String {
    if !log_bf.is_finite() {
        return "inf".to_string();
    }
    if log_bf.abs() >= std::f64::consts::LN_10 * 3.0 {
        return format!("1e{:+.1}", log_bf / std::f64::consts::LN_10);
    }
    format_three_significant(log_bf.exp())
}

fn format_three_significant(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    let decimals = if abs >= 100.0 {
        0
    } else if abs >= 10.0 {
        1
    } else if abs >= 1.0 {
        2
    } else {
        let exponent = abs.log10().floor();
        (-exponent as i32 + 2).max(0) as usize
    };
    let mut formatted = format!("{value:.decimals$}");
    if formatted.contains('.') {
        while formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.ends_with('.') {
            formatted.pop();
        }
    }
    formatted
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
    let x_values = x.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let n_rows = x_values.nrows();
    let n_outputs = y_values.ncols();
    let n_coefficients = penalty_values.nrows();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let result = detach_py_result(py, "gaussian_reml_fit", move || {
        let gated_x;
        let fit_x = if let Some(by_arr) = by_values.as_ref() {
            gated_x = apply_by_gate(x_values.view(), by_arr.view(), by_start_col)?;
            gated_x.view()
        } else {
            x_values.view()
        };
        match gaussian_reml_multi_closed_form_with_cache(
            fit_x,
            y_values.view(),
            penalty_values.view(),
            weight_values.as_ref().map(|w| w.view()),
            init_lambda,
            None,
        ) {
            Ok(fit) => Ok(Some(fit)),
            Err(EstimationError::ModelIsIllConditioned { .. }) => Ok(None),
            Err(err) => Err(err.to_string()),
        }
    })?;
    let out = PyDict::new(py);
    match result {
        Some(fit) => {
            set_ok_gaussian_reml_items(py, &out, fit)?;
        }
        None => {
            set_degenerate_gaussian_reml_items(py, &out, n_rows, n_outputs, n_coefficients)?;
        }
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
    grad_edf = 0.0,
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
    grad_edf: f64,
    forward_state: Option<&Bound<'py, PyDict>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let x_values = x.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let grad_coefficients_values = grad_coefficients.as_ref().map(|g| g.as_array().to_owned());
    let grad_fitted_values = grad_fitted.as_ref().map(|g| g.as_array().to_owned());
    let forward_fit = forward_state
        .map(gaussian_reml_fit_state_from_pydict)
        .transpose()
        .map_err(py_value_error)?;
    let (grad_x, grad_by, grad_y, grad_penalty, grad_weights) =
        detach_py_result(py, "gaussian_reml_fit_backward", move || {
            let gated_x;
            let fit_x = if let Some(by_arr) = by_values.as_ref() {
                gated_x = apply_by_gate(x_values.view(), by_arr.view(), by_start_col)?;
                gated_x.view()
            } else {
                x_values.view()
            };
            let backward = if let Some(fit) = forward_fit.as_ref() {
                gaussian_reml_multi_closed_form_backward_from_fit(
                    fit_x,
                    y_values.view(),
                    penalty_values.view(),
                    weight_values.as_ref().map(|w| w.view()),
                    fit,
                    grad_lambda,
                    grad_coefficients_values.as_ref().map(|g| g.view()),
                    grad_fitted_values.as_ref().map(|g| g.view()),
                    grad_reml_score,
                    grad_edf,
                )
            } else {
                gaussian_reml_multi_closed_form_backward(
                    fit_x,
                    y_values.view(),
                    penalty_values.view(),
                    weight_values.as_ref().map(|w| w.view()),
                    init_lambda,
                    grad_lambda,
                    grad_coefficients_values.as_ref().map(|g| g.view()),
                    grad_fitted_values.as_ref().map(|g| g.view()),
                    grad_reml_score,
                    grad_edf,
                )
            }
            .map_err(|err| err.to_string())?;
            let grad_x_raw = backward.grad_x;
            let (grad_x, grad_by) = if let Some(by_arr) = by_values.as_ref() {
                let (grad_x, grad_by) = apply_by_gate_backward(
                    x_values.view(),
                    by_arr.view(),
                    by_start_col,
                    grad_x_raw.view(),
                )?;
                (grad_x, Some(grad_by))
            } else {
                (grad_x_raw, None)
            };
            Ok((
                grad_x,
                grad_by,
                backward.grad_y,
                backward.grad_penalty,
                backward.grad_weights,
            ))
        })?;

    let out = PyDict::new(py);
    out.set_item("grad_x", grad_x.into_pyarray(py))?;
    out.set_item("grad_y", grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", grad_penalty.into_pyarray(py))?;
    out.set_item("grad_weights", grad_weights.into_pyarray(py))?;
    if let Some(grad_by) = grad_by {
        out.set_item("grad_by", grad_by.into_pyarray(py))?;
    } else {
        out.set_item("grad_by", py.None())?;
    }
    Ok(out.unbind())
}

#[pyfunction(signature = (headers, rows, formula, y, config_json = None, fisher_rao_w = None))]
fn gaussian_reml_fit_formula_table<'py>(
    py: Python<'py>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    y: PyReadonlyArray2<'py, f64>,
    config_json: Option<String>,
    fisher_rao_w: Option<PyReadonlyArray3<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    let y_values = y.as_array().to_owned();
    let fisher_values = fisher_rao_w.as_ref().map(|w| w.as_array().to_owned());
    let result = detach_py_result(py, "gaussian_reml_fit_formula_table", move || {
        gaussian_reml_fit_formula_table_impl(
            headers,
            rows,
            formula,
            y_values.view(),
            config_json.as_deref(),
            fisher_values.as_ref().map(|w| w.view()),
        )
    })?;
    gaussian_reml_result_to_pydict(py, result)
}

/// Multi-block Gaussian REML forward fit with per-smooth λ_k.
///
/// Programmatic (formula-API-bypass) entry into the same joint multi-smooth
/// REML driver used by `gamfit.fit(data, "y ~ s(x1) + s(x2)")`: concatenates
/// per-smooth design blocks into a global X, builds a `BlockwisePenalty`
/// list (one entry per smooth), and runs `fit_gam` with one λ_k per block.
///
/// Inputs:
/// - `designs`: list of per-smooth design blocks `(N, K_k)`.
/// - `penalties`: list of per-smooth penalty blocks `(K_k, K_k)`.
/// - `y`: response `(N, 1)`. Multi-output `(N, D>1)` is unsupported here.
/// - `weights`: optional row weights `(N,)`.
/// - `init_rhos`: optional warm-start log-λ vector of length F.
///
/// Returns a dict with `coefficients` `(P_total, 1)`, `fitted` `(N, 1)`,
/// `lambdas` `(F,)`, `reml_score` (scalar), `edf` `(F,)`, `col_offsets`
/// `(F+1,)`.
#[pyfunction(signature = (
    designs,
    penalties,
    y,
    weights = None,
    init_rhos = None
))]
fn gaussian_reml_fit_blocks_forward<'py>(
    py: Python<'py>,
    designs: Vec<PyReadonlyArray2<'py, f64>>,
    penalties: Vec<PyReadonlyArray2<'py, f64>>,
    y: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_rhos: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    if designs.is_empty() {
        return Err(py_value_error(
            "gaussian_reml_fit_blocks_forward requires at least one block".to_string(),
        ));
    }
    if designs.len() != penalties.len() {
        return Err(py_value_error(format!(
            "designs and penalties must have equal length; got {} vs {}",
            designs.len(),
            penalties.len(),
        )));
    }

    let n_rows = designs[0].as_array().nrows();
    let mut col_offsets: Vec<usize> = vec![0];
    for (i, d) in designs.iter().enumerate() {
        let view = d.as_array();
        if view.nrows() != n_rows {
            return Err(py_value_error(format!(
                "designs[{}].nrows={} does not match designs[0].nrows={}",
                i,
                view.nrows(),
                n_rows,
            )));
        }
        if let Some(((row, col), value)) = view.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(py_value_error(format!(
                "designs[{i}][{row},{col}] must be finite; got {value}"
            )));
        }
        col_offsets.push(col_offsets[i] + view.ncols());
    }
    let p_total = *col_offsets.last().unwrap();
    let mut joint_x = Array2::<f64>::zeros((n_rows, p_total));
    for (i, d) in designs.iter().enumerate() {
        joint_x
            .slice_mut(s![.., col_offsets[i]..col_offsets[i + 1]])
            .assign(&d.as_array());
    }

    let mut s_list: Vec<gam::smooth::BlockwisePenalty> = Vec::with_capacity(designs.len());
    for (i, p) in penalties.iter().enumerate() {
        let pv = p.as_array();
        let k = col_offsets[i + 1] - col_offsets[i];
        if pv.nrows() != k || pv.ncols() != k {
            return Err(py_value_error(format!(
                "penalties[{}] shape {}x{} does not match design block size {}",
                i,
                pv.nrows(),
                pv.ncols(),
                k,
            )));
        }
        if let Some(((row, col), value)) = pv.indexed_iter().find(|(_, value)| !value.is_finite()) {
            return Err(py_value_error(format!(
                "penalties[{i}][{row},{col}] must be finite; got {value}"
            )));
        }
        s_list.push(gam::smooth::BlockwisePenalty::new(
            col_offsets[i]..col_offsets[i + 1],
            pv.to_owned(),
        ));
    }

    let y_arr = y.as_array();
    if y_arr.nrows() != n_rows {
        return Err(py_value_error(format!(
            "y.nrows={} does not match design N={}",
            y_arr.nrows(),
            n_rows,
        )));
    }
    if y_arr.ncols() != 1 {
        return Err(py_value_error(format!(
            "gaussian_reml_fit_blocks_forward requires y of shape (N, 1); got (N, {})",
            y_arr.ncols(),
        )));
    }
    if let Some(((row, col), value)) = y_arr.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(py_value_error(format!(
            "y[{row},{col}] must be finite; got {value}"
        )));
    }
    let y_col: ndarray::Array1<f64> = y_arr.column(0).to_owned();

    let weights_owned: ndarray::Array1<f64> = match weights.as_ref() {
        Some(w) => {
            let wa = w.as_array();
            if wa.len() != n_rows {
                return Err(py_value_error(format!(
                    "weights.len={} does not match N={}",
                    wa.len(),
                    n_rows,
                )));
            }
            if let Some((row, value)) = wa
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite() || **value < 0.0)
            {
                return Err(py_value_error(format!(
                    "weights[{row}] must be finite and non-negative; got {value}"
                )));
            }
            wa.to_owned()
        }
        None => ndarray::Array1::from_elem(n_rows, 1.0),
    };
    let heuristic_owned: Option<Vec<f64>> = match init_rhos.as_ref() {
        Some(r) => {
            let rv = r.as_array();
            if rv.len() != designs.len() {
                return Err(py_value_error(format!(
                    "init_rhos.len={} does not match F={}",
                    rv.len(),
                    designs.len(),
                )));
            }
            if let Some((block, value)) =
                rv.iter().enumerate().find(|(_, value)| !value.is_finite())
            {
                return Err(py_value_error(format!(
                    "init_rhos[{block}] must be finite; got {value}"
                )));
            }
            let lambdas: Vec<f64> = rv.iter().map(|rho| rho.exp()).collect();
            if let Some((block, value)) = lambdas
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite() || **value <= 0.0)
            {
                return Err(py_value_error(format!(
                    "exp(init_rhos[{block}]) must be finite and positive; got {value}"
                )));
            }
            Some(lambdas)
        }
        None => None,
    };

    let offset_zero = Array1::<f64>::zeros(n_rows);
    let opts = gam::estimate::FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 200,
        tol: 1.0e-9,
        nullspace_dims: vec![0; s_list.len()],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let joint_x_for_fit = joint_x.clone();
    let fit = detach_estimation_result(py, "gaussian_reml_fit_blocks_forward", move || {
        let heuristic_slice = heuristic_owned.as_ref().map(|values| values.as_slice());
        gam::estimate::fit_gamwith_heuristic_lambdas(
            joint_x_for_fit,
            y_col.view(),
            weights_owned.view(),
            offset_zero.view(),
            &s_list,
            heuristic_slice,
            LikelihoodSpec::new(ResponseFamily::Gaussian, InverseLink::Standard(LinkFunction::Identity)),
            &opts,
        )
    })?;

    let lambdas = fit.lambdas.clone();
    if let Some((block, value)) = lambdas
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || **value <= 0.0)
    {
        return Err(py_value_error(format!(
            "fitted lambda[{block}] must be finite and positive; got {value}"
        )));
    }
    let edf_vec = fit
        .inference
        .as_ref()
        .map(|inf| inf.edf_by_block.clone())
        .unwrap_or_else(|| vec![0.0; lambdas.len()]);
    let edf_arr = if edf_vec.len() == lambdas.len() {
        Array1::from_vec(edf_vec)
    } else {
        Array1::zeros(lambdas.len())
    };
    let coefficients_2d = fit.beta.clone().insert_axis(ndarray::Axis(1));
    let fitted_2d = joint_x.dot(&fit.beta).insert_axis(ndarray::Axis(1));
    let log_lambdas_arr = lambdas.mapv(|value| value.max(1.0e-300).ln());

    let out = PyDict::new(py);
    out.set_item("coefficients", coefficients_2d.into_pyarray(py))?;
    out.set_item("fitted", fitted_2d.into_pyarray(py))?;
    out.set_item("lambdas", lambdas.into_pyarray(py))?;
    out.set_item("log_lambdas", log_lambdas_arr.into_pyarray(py))?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("edf", edf_arr.into_pyarray(py))?;
    out.set_item(
        "col_offsets",
        ndarray::Array1::from_iter(col_offsets.into_iter().map(|v| v as u64)).into_pyarray(py),
    )?;
    Ok(out.unbind())
}

struct GaussianRemlBlocksBackwardAnalytic {
    grad_designs: Vec<Array2<f64>>,
    grad_penalties: Vec<Array2<f64>>,
    grad_y: Array2<f64>,
    grad_weights: Array1<f64>,
}

fn identity_matrix(n: usize) -> Array2<f64> {
    let mut eye = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        eye[[i, i]] = 1.0;
    }
    eye
}

fn symmetrized_matrix(input: &Array2<f64>) -> Array2<f64> {
    let n = input.nrows();
    let mut out = input.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}

fn symmetrize_in_place(input: &mut Array2<f64>) {
    let n = input.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (input[[i, j]] + input[[j, i]]);
            input[[i, j]] = avg;
            input[[j, i]] = avg;
        }
    }
}

fn block_penalty_rank_and_pinv(
    penalty: &Array2<f64>,
) -> Result<(usize, Array2<f64>), EstimationError> {
    let (eigs, vecs) = penalty.to_owned().eigh(Side::Lower).map_err(|_| {
        EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        }
    })?;
    let max_abs = eigs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let tol = (1.0e-10 * max_abs).max(1.0e-14);
    let mut rank = 0_usize;
    let mut scaled = Array2::<f64>::zeros(vecs.dim());
    for col in 0..eigs.len() {
        if eigs[col] > tol {
            rank += 1;
            for row in 0..vecs.nrows() {
                scaled[[row, col]] = vecs[[row, col]] / eigs[col];
            }
        }
    }
    Ok((rank, scaled.dot(&vecs.t())))
}

fn invert_spd_with_ridge(
    matrix: &Array2<f64>,
    ridge_rel: f64,
) -> Result<Array2<f64>, EstimationError> {
    let n = matrix.nrows();
    let eye = identity_matrix(n);
    let scale = (0..n).map(|i| matrix[[i, i]].abs()).fold(1.0_f64, f64::max);
    let ridges = [0.0, ridge_rel, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4];
    for rel in ridges {
        let mut candidate = matrix.clone();
        if rel > 0.0 {
            for i in 0..n {
                candidate[[i, i]] += rel * scale;
            }
        }
        if let Ok(chol) = candidate.cholesky(Side::Lower) {
            return Ok(chol.solve_mat(&eye));
        }
    }
    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

fn solve_symmetric_vector_with_floor(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    ridge_rel: f64,
) -> Result<Array1<f64>, EstimationError> {
    let n = matrix.nrows();
    let sym = symmetrized_matrix(matrix);
    let (eigs, vecs) =
        sym.eigh(Side::Lower)
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let max_eig = eigs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let floor = (ridge_rel * max_eig.max(1.0)).max(1.0e-12);
    let projected = vecs.t().dot(rhs);
    let mut scaled = Array1::<f64>::zeros(n);
    for i in 0..n {
        let denom = if eigs[i].abs() >= floor {
            eigs[i]
        } else if eigs[i].is_sign_negative() {
            -floor
        } else {
            floor
        };
        scaled[i] = projected[i] / denom;
    }
    let out = vecs.dot(&scaled);
    if out.iter().all(|value| value.is_finite()) {
        Ok(out)
    } else {
        Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }
}

fn trace_product(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> f64 {
    let mut value = 0.0;
    for i in 0..left.nrows() {
        for j in 0..left.ncols() {
            value += left[[i, j]] * right[[j, i]];
        }
    }
    value
}

#[allow(clippy::too_many_arguments)]
fn gaussian_reml_fit_blocks_backward_analytic(
    designs: &[Array2<f64>],
    penalties_raw: &[Array2<f64>],
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    rhos: &[f64],
    grad_coefficients: Option<ArrayView2<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_lambdas: Option<ArrayView1<'_, f64>>,
    grad_log_lambdas: Option<ArrayView1<'_, f64>>,
    grad_reml_score: f64,
    grad_edf: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlBlocksBackwardAnalytic, EstimationError> {
    let n = y.len();
    let f_blocks = designs.len();
    let mut offsets = Vec::with_capacity(f_blocks + 1);
    offsets.push(0_usize);
    for design in designs {
        offsets.push(offsets.last().copied().unwrap() + design.ncols());
    }
    let p_total = *offsets.last().unwrap();

    if rhos.len() != f_blocks {
        return Err(EstimationError::InvalidInput(format!(
            "log_lambdas length mismatch: expected {f_blocks}, got {}",
            rhos.len()
        )));
    }
    if let Some(gc) = grad_coefficients {
        if gc.dim() != (p_total, 1) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_coefficients shape mismatch: expected {}x1, got {}x{}",
                p_total,
                gc.nrows(),
                gc.ncols()
            )));
        }
    }
    if let Some(gf) = grad_fitted {
        if gf.dim() != (n, 1) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_fitted shape mismatch: expected {}x1, got {}x{}",
                n,
                gf.nrows(),
                gf.ncols()
            )));
        }
    }
    if !grad_reml_score.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "grad_reml_score must be finite; got {grad_reml_score}"
        )));
    }
    if let Some(vec) = grad_lambdas {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_lambdas length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(vec) = grad_log_lambdas {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_log_lambdas length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(vec) = grad_edf {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_edf length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(gc) = grad_coefficients {
        if let Some(((row, col), value)) = gc.indexed_iter().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_coefficients[{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some(gf) = grad_fitted {
        if let Some(((row, col), value)) = gf.indexed_iter().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_fitted[{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_lambdas {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_lambdas[{block}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_log_lambdas {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_log_lambdas[{block}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_edf {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_edf[{block}] must be finite; got {value}"
            )));
        }
    }
    for (block, design) in designs.iter().enumerate() {
        if let Some(((row, col), value)) =
            design.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "designs[{block}][{row},{col}] must be finite; got {value}"
            )));
        }
    }
    for (block, penalty) in penalties_raw.iter().enumerate() {
        if let Some(((row, col), value)) =
            penalty.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "penalties[{block}][{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some((row, value)) = y.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "y[{row}] must be finite; got {value}"
        )));
    }
    if let Some((row, value)) = weights
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || **value < 0.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "weights[{row}] must be finite and non-negative; got {value}"
        )));
    }

    let mut z = Array2::<f64>::zeros((n, p_total));
    for k in 0..f_blocks {
        z.slice_mut(s![.., offsets[k]..offsets[k + 1]])
            .assign(&designs[k]);
    }

    let penalties: Vec<Array2<f64>> = penalties_raw.iter().map(symmetrized_matrix).collect();
    let mut ranks = Vec::with_capacity(f_blocks);
    let mut pinvs = Vec::with_capacity(f_blocks);
    for penalty in &penalties {
        let (rank, pinv) = block_penalty_rank_and_pinv(penalty)?;
        ranks.push(rank);
        pinvs.push(pinv);
    }

    let lambdas = Array1::from_iter(rhos.iter().map(|rho| rho.exp()));
    if let Some((block, lambda)) = lambdas
        .iter()
        .enumerate()
        .find(|(_, lambda)| !lambda.is_finite() || **lambda <= 0.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "exp(log_lambdas[{block}]) must be finite and positive; got {lambda}"
        )));
    }
    let mut k_matrix = fast_xt_diag_x(&z.view(), &weights);
    for block in 0..f_blocks {
        let lambda = lambdas[block];
        for local_i in 0..penalties[block].nrows() {
            let global_i = offsets[block] + local_i;
            for local_j in 0..penalties[block].ncols() {
                let global_j = offsets[block] + local_j;
                k_matrix[[global_i, global_j]] += lambda * penalties[block][[local_i, local_j]];
            }
        }
    }
    let r = invert_spd_with_ridge(&k_matrix, 0.0)?;

    let mut xtwy = Array1::<f64>::zeros(p_total);
    for row in 0..n {
        let wy = weights[row] * y[row];
        for col in 0..p_total {
            xtwy[col] += z[[row, col]] * wy;
        }
    }
    let beta = r.dot(&xtwy);
    let fitted = z.dot(&beta);
    if let Some((col, value)) = beta
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "solved coefficient {col} is non-finite: {value}"
        )));
    }
    let residual = &y.to_owned() - &fitted;
    let weighted_residual = &residual * &weights.to_owned();
    let ywy = y
        .iter()
        .zip(weights.iter())
        .map(|(&yi, &wi)| wi * yi * yi)
        .sum::<f64>();
    let q_raw = ywy - xtwy.dot(&beta);
    if !q_raw.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML residual quadratic form must be finite; got {q_raw}"
        )));
    }
    let q = q_raw.max(1.0e-300);
    let nullity = penalties
        .iter()
        .zip(ranks.iter())
        .map(|(penalty, rank)| penalty.nrows().saturating_sub(*rank))
        .sum::<usize>();
    let nu = n as f64 - nullity as f64;
    if !(nu.is_finite() && nu > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML residual degrees of freedom must be positive; got {nu}"
        )));
    }
    let tau = nu / q;
    let tau_q = -nu / (q * q);
    if !(tau.is_finite() && tau_q.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML scale derivatives are non-finite: tau={tau}, tau_q={tau_q}"
        )));
    }

    let mut grad_z = Array2::<f64>::zeros((n, p_total));
    let mut g_kernel = Array2::<f64>::zeros((p_total, p_total));
    let mut h_kernel = Array1::<f64>::zeros(p_total);
    let mut q_kernel = 0.0_f64;
    let mut j_blocks: Vec<Array2<f64>> = penalties
        .iter()
        .map(|p| Array2::<f64>::zeros(p.dim()))
        .collect();

    let mut beta_tilde = Array1::<f64>::zeros(p_total);
    if let Some(gc) = grad_coefficients {
        beta_tilde += &gc.column(0).to_owned();
    }
    if let Some(gf) = grad_fitted {
        let gf_col = gf.column(0).to_owned();
        beta_tilde += &z.t().dot(&gf_col);
        for row in 0..n {
            for col in 0..p_total {
                grad_z[[row, col]] += gf_col[row] * beta[col];
            }
        }
    }

    // Generic downstream losses that explicitly seed beta_hat or fitted
    // values cannot use the REML envelope shortcut. Route those seeds through
    // the fixed-rho KKT adjoint K u = beta_tilde before differentiating
    // designs, penalties, y, weights, and rho.
    let u = r.dot(&beta_tilde);
    h_kernel += &u;
    for i in 0..p_total {
        for j in 0..p_total {
            g_kernel[[i, j]] -= 0.5 * (beta[i] * u[j] + u[i] * beta[j]);
        }
    }

    let mut alpha = Array1::<f64>::zeros(f_blocks);
    if let Some(gl) = grad_lambdas {
        for block in 0..f_blocks {
            alpha[block] += gl[block] * lambdas[block];
        }
    }
    if let Some(grho) = grad_log_lambdas {
        alpha += &grho.to_owned();
    }

    let mut p_betas = Vec::with_capacity(f_blocks);
    let mut m_vectors = Vec::with_capacity(f_blocks);
    let mut rp_matrices = Vec::with_capacity(f_blocks);
    let mut rpr_matrices = Vec::with_capacity(f_blocks);
    let mut b_values = Array1::<f64>::zeros(f_blocks);
    let mut t_values = Array1::<f64>::zeros(f_blocks);

    for block in 0..f_blocks {
        let start = offsets[block];
        let end = offsets[block + 1];
        let beta_k = beta.slice(s![start..end]).to_owned();
        let s_beta = penalties[block].dot(&beta_k);
        let lambda = lambdas[block];
        let lambda_s_beta = s_beta.mapv(|value| lambda * value);
        let mut p_beta = Array1::<f64>::zeros(p_total);
        for local_i in 0..(end - start) {
            p_beta[start + local_i] = lambda_s_beta[local_i];
        }
        let weighted_penalty = penalties[block].mapv(|value| lambda * value);
        let rp_block = r.slice(s![.., start..end]).dot(&weighted_penalty);
        let mut rp = Array2::<f64>::zeros((p_total, p_total));
        rp.slice_mut(s![.., start..end]).assign(&rp_block);
        let rpr = rp_block.dot(&r.slice(s![start..end, ..]));
        let m = r.slice(s![.., start..end]).dot(&lambda_s_beta);
        b_values[block] = beta.dot(&p_beta);
        t_values[block] = (0..(end - start))
            .map(|local_i| rp_block[[start + local_i, local_i]])
            .sum::<f64>();
        alpha[block] -= u.dot(&p_beta);
        p_betas.push(p_beta);
        m_vectors.push(m);
        rp_matrices.push(rp);
        rpr_matrices.push(rpr);
    }

    if grad_reml_score != 0.0 {
        q_kernel += 0.5 * grad_reml_score * tau;
        g_kernel += &(r.clone() * (0.5 * grad_reml_score));
        for block in 0..f_blocks {
            j_blocks[block] -= &(pinvs[block].clone() * (0.5 * grad_reml_score / lambdas[block]));
        }
    }

    let mut trace_pairs = Array2::<f64>::zeros((f_blocks, f_blocks));
    for i in 0..f_blocks {
        for j in 0..f_blocks {
            trace_pairs[[i, j]] = trace_product(rp_matrices[i].view(), rp_matrices[j].view());
        }
    }

    if let Some(ge) = grad_edf {
        for edf_block in 0..f_blocks {
            let scale = ge[edf_block];
            if scale == 0.0 {
                continue;
            }
            let start = offsets[edf_block];
            let end = offsets[edf_block + 1];
            g_kernel += &(rpr_matrices[edf_block].clone() * scale);
            j_blocks[edf_block] -= &(r.slice(s![start..end, start..end]).to_owned() * scale);
            for rho_block in 0..f_blocks {
                alpha[rho_block] += scale * trace_pairs[[edf_block, rho_block]];
                if rho_block == edf_block {
                    alpha[rho_block] -= scale * t_values[edf_block];
                }
            }
        }
    }

    if let Some((block, value)) = alpha
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "rho adjoint seed for block {block} is non-finite: {value}"
        )));
    }

    if alpha.iter().any(|value| *value != 0.0) {
        let mut outer_h = Array2::<f64>::zeros((f_blocks, f_blocks));
        for k in 0..f_blocks {
            for j in 0..f_blocks {
                let beta_pk_r_pj_beta = p_betas[k].dot(&m_vectors[j]);
                outer_h[[k, j]] = 0.5 * trace_pairs[[k, j]] + tau * beta_pk_r_pj_beta
                    - if k == j {
                        0.5 * (t_values[k] + tau * b_values[k])
                    } else {
                        0.0
                    }
                    - 0.5 * tau_q * b_values[k] * b_values[j];
            }
        }
        // `outer_h` is the Jacobian of the negative profiled REML estimating
        // equation. Preserve signed curvature directions while flooring
        // near-zero modes; flipping negative eigenvalues would change the VJP.
        symmetrize_in_place(&mut outer_h);
        if let Some(((row, col), value)) =
            outer_h.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "outer rho curvature entry ({row},{col}) is non-finite: {value}"
            )));
        }
        let rho_adj = solve_symmetric_vector_with_floor(&outer_h, &alpha, 1.0e-10)?;
        if let Some((block, value)) = rho_adj
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "outer rho adjoint for block {block} is non-finite: {value}"
            )));
        }
        let weighted_b_sum = rho_adj
            .iter()
            .zip(b_values.iter())
            .map(|(&zk, &bk)| zk * bk)
            .sum::<f64>();
        q_kernel += 0.5 * tau_q * weighted_b_sum;
        for block in 0..f_blocks {
            let zk = rho_adj[block];
            if zk == 0.0 {
                continue;
            }
            g_kernel -= &(rpr_matrices[block].clone() * (0.5 * zk));
            let m = &m_vectors[block];
            for i in 0..p_total {
                h_kernel[i] += tau * zk * m[i];
                for j in 0..p_total {
                    g_kernel[[i, j]] -= 0.5 * tau * zk * (beta[i] * m[j] + m[i] * beta[j]);
                }
            }
            let start = offsets[block];
            let end = offsets[block + 1];
            j_blocks[block] += &(r.slice(s![start..end, start..end]).to_owned() * (0.5 * zk));
            for i in 0..(end - start) {
                for j in 0..(end - start) {
                    j_blocks[block][[i, j]] += 0.5 * tau * zk * beta[start + i] * beta[start + j];
                }
            }
        }
    }

    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += -2.0 * q_kernel * weighted_residual[row] * beta[col];
        }
    }
    let zg = z.dot(&g_kernel);
    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += 2.0 * weights[row] * zg[[row, col]];
        }
    }
    let wy = y.to_owned() * &weights.to_owned();
    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += wy[row] * h_kernel[col];
        }
    }

    let mut grad_y = Array2::<f64>::zeros((n, 1));
    let zh = z.dot(&h_kernel);
    for row in 0..n {
        grad_y[[row, 0]] = 2.0 * q_kernel * weighted_residual[row] + weights[row] * zh[row];
    }

    let mut grad_weights = Array1::<f64>::zeros(n);
    for row in 0..n {
        let diag_zgz = (0..p_total)
            .map(|col| z[[row, col]] * zg[[row, col]])
            .sum::<f64>();
        grad_weights[row] = q_kernel * residual[row] * residual[row] + diag_zgz + y[row] * zh[row];
    }

    let mut grad_penalties = Vec::with_capacity(f_blocks);
    for block in 0..f_blocks {
        let start = offsets[block];
        let end = offsets[block + 1];
        let mut local = g_kernel.slice(s![start..end, start..end]).to_owned();
        for i in 0..(end - start) {
            for j in 0..(end - start) {
                local[[i, j]] += q_kernel * beta[start + i] * beta[start + j];
            }
        }
        local += &j_blocks[block];
        local *= lambdas[block];
        symmetrize_in_place(&mut local);
        grad_penalties.push(local);
    }

    let mut grad_designs = Vec::with_capacity(f_blocks);
    for block in 0..f_blocks {
        grad_designs.push(
            grad_z
                .slice(s![.., offsets[block]..offsets[block + 1]])
                .to_owned(),
        );
    }

    Ok(GaussianRemlBlocksBackwardAnalytic {
        grad_designs,
        grad_penalties,
        grad_y,
        grad_weights,
    })
}

/// Analytic backward for the multi-block per-smooth-λ Gaussian REML forward.
///
/// Computes VJPs of (coefficients, fitted, lambdas, log_lambdas, reml_score,
/// edf) back to (design_blocks, penalty_blocks, y, weights). The VJP is
/// assembled at the converged log-λ vector: fixed-ρ β/fitted/profiled-REML/EDF
/// terms are accumulated first, then the smoothing-parameter sensitivity is
/// routed through the F×F profiled REML score Hessian from the implicit optimum.
#[pyfunction(signature = (
    designs,
    penalties,
    y,
    weights,
    log_lambdas,
    grad_coefficients = None,
    grad_fitted = None,
    grad_lambdas = None,
    grad_log_lambdas = None,
    grad_reml_score = 0.0,
    grad_edf = None,
))]
fn gaussian_reml_fit_blocks_backward<'py>(
    py: Python<'py>,
    designs: Vec<PyReadonlyArray2<'py, f64>>,
    penalties: Vec<PyReadonlyArray2<'py, f64>>,
    y: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    log_lambdas: PyReadonlyArray1<'py, f64>,
    grad_coefficients: Option<PyReadonlyArray2<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_lambdas: Option<PyReadonlyArray1<'py, f64>>,
    grad_log_lambdas: Option<PyReadonlyArray1<'py, f64>>,
    grad_reml_score: f64,
    grad_edf: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    if designs.is_empty() {
        return Err(py_value_error(
            "gaussian_reml_fit_blocks_backward requires at least one block".to_string(),
        ));
    }
    if designs.len() != penalties.len() {
        return Err(py_value_error(format!(
            "designs and penalties must have equal length; got {} vs {}",
            designs.len(),
            penalties.len(),
        )));
    }

    let n_rows = designs[0].as_array().nrows();
    let designs_owned: Vec<Array2<f64>> = designs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let v = d.as_array();
            if v.nrows() != n_rows {
                return Err(py_value_error(format!(
                    "designs[{}].nrows={} does not match designs[0].nrows={}",
                    i,
                    v.nrows(),
                    n_rows,
                )));
            }
            Ok(v.to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let penalties_owned: Vec<Array2<f64>> = penalties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let v = p.as_array();
            let k = designs_owned[i].ncols();
            if v.nrows() != k || v.ncols() != k {
                return Err(py_value_error(format!(
                    "penalties[{}] shape {}x{} does not match design block size {}",
                    i,
                    v.nrows(),
                    v.ncols(),
                    k,
                )));
            }
            Ok(v.to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let y_arr = y.as_array();
    if y_arr.nrows() != n_rows {
        return Err(py_value_error(format!(
            "y.nrows={} does not match design N={}",
            y_arr.nrows(),
            n_rows,
        )));
    }
    if y_arr.ncols() != 1 {
        return Err(py_value_error(format!(
            "gaussian_reml_fit_blocks_backward requires y of shape (N, 1); got (N, {})",
            y_arr.ncols(),
        )));
    }
    let y_owned: Array1<f64> = y_arr.column(0).to_owned();

    let weights_owned: Array1<f64> = match weights.as_ref() {
        Some(w) => {
            let wa = w.as_array();
            if wa.len() != n_rows {
                return Err(py_value_error(format!(
                    "weights.len={} does not match N={}",
                    wa.len(),
                    n_rows,
                )));
            }
            if wa.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err(py_value_error(
                    "weights must contain finite non-negative values".to_string(),
                ));
            }
            wa.to_owned()
        }
        None => Array1::from_elem(n_rows, 1.0),
    };

    let rhos_view = log_lambdas.as_array();
    if rhos_view.len() != designs_owned.len() {
        return Err(py_value_error(format!(
            "log_lambdas.len={} does not match F={}",
            rhos_view.len(),
            designs_owned.len(),
        )));
    }
    let init_rhos: Vec<f64> = rhos_view.iter().copied().collect();
    if init_rhos.iter().any(|value| !value.is_finite()) {
        return Err(py_value_error(
            "log_lambdas must contain only finite values".to_string(),
        ));
    }

    let grad_coef_owned = grad_coefficients.as_ref().map(|g| g.as_array().to_owned());
    let grad_fitted_owned = grad_fitted.as_ref().map(|g| g.as_array().to_owned());
    let grad_lam_owned = grad_lambdas.as_ref().map(|g| g.as_array().to_owned());
    let grad_log_lam_owned = grad_log_lambdas.as_ref().map(|g| g.as_array().to_owned());
    let grad_edf_owned = grad_edf.as_ref().map(|g| g.as_array().to_owned());

    let designs_for_thread = designs_owned.clone();
    let penalties_for_thread = penalties_owned.clone();
    let y_for_thread = y_owned.clone();
    let weights_for_thread = weights_owned.clone();
    let init_rhos_for_thread = init_rhos.clone();

    let backward = detach_estimation_result(py, "gaussian_reml_fit_blocks_backward", move || {
        gaussian_reml_fit_blocks_backward_analytic(
            &designs_for_thread,
            &penalties_for_thread,
            y_for_thread.view(),
            weights_for_thread.view(),
            init_rhos_for_thread.as_slice(),
            grad_coef_owned.as_ref().map(|a| a.view()),
            grad_fitted_owned.as_ref().map(|a| a.view()),
            grad_lam_owned.as_ref().map(|a| a.view()),
            grad_log_lam_owned.as_ref().map(|a| a.view()),
            grad_reml_score,
            grad_edf_owned.as_ref().map(|a| a.view()),
        )
    })?;

    let out = PyDict::new(py);
    let grad_designs_py: Vec<Bound<'py, PyArray2<f64>>> = backward
        .grad_designs
        .into_iter()
        .map(|a| a.into_pyarray(py))
        .collect();
    let grad_penalties_py: Vec<Bound<'py, PyArray2<f64>>> = backward
        .grad_penalties
        .into_iter()
        .map(|a| a.into_pyarray(py))
        .collect();
    out.set_item("grad_designs", grad_designs_py)?;
    out.set_item("grad_penalties", grad_penalties_py)?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    Ok(out.unbind())
}

/// Constrained Gaussian REML forward fit with a single penalty block and an
/// optional linear inequality system `A·β ≤ b`.
///
/// Wraps the same constrained PIRLS+REML driver (`fit_gam` with
/// `FitOptions.linear_constraints`) that backs the formula-API shape
/// constraints. Forward-only: no analytic VJP through the active-set
/// inner solver is exposed here (the BUG-3 tangent-projection backward
/// is implemented inside the REML driver but not yet plumbed out as a
/// reusable Python VJP — see `gaussian_reml_fit_blocks_forward` for the
/// equivalent forward-only contract).
///
/// Inputs:
/// * `x` — design matrix `(N, M)` (single block).
/// * `y` — response `(N, 1)`.
/// * `penalty` — `(M, M)` smoothing penalty for the single block.
/// * `weights` — optional row weights `(N,)`; defaults to ones.
/// * `init_log_lambda` — optional scalar warm-start in log-λ.
/// * `a_inequality` — `(R, M)` inequality matrix; pass an empty (0×M)
///   array (or `None`) for the unconstrained case.
/// * `b_inequality` — `(R,)` inequality RHS; same length as `a_inequality.nrows()`.
///
/// Outputs (dict): `coefficients (M, 1)`, `fitted (N, 1)`,
/// `lambda` (scalar), `log_lambda` (scalar), `reml_score` (scalar),
/// `edf` (scalar), `active_indices` (`(K,)` uint64 row indices of `A` at
/// the converged β).
#[pyfunction(signature = (
    x,
    y,
    penalty,
    weights = None,
    init_log_lambda = None,
    a_inequality = None,
    b_inequality = None,
))]
fn gaussian_reml_fit_with_constraints_forward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_log_lambda: Option<f64>,
    a_inequality: Option<PyReadonlyArray2<'py, f64>>,
    b_inequality: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    let x_view = x.as_array();
    let y_view = y.as_array();
    let penalty_view = penalty.as_array();

    let n_rows = x_view.nrows();
    let p_cols = x_view.ncols();

    if y_view.nrows() != n_rows {
        return Err(py_value_error(format!(
            "gaussian_reml_fit_with_constraints_forward: y has {} rows but X has {}",
            y_view.nrows(),
            n_rows,
        )));
    }
    if y_view.ncols() != 1 {
        return Err(py_value_error(format!(
            "gaussian_reml_fit_with_constraints_forward requires y of shape (N, 1); got (N, {})",
            y_view.ncols(),
        )));
    }
    if penalty_view.nrows() != p_cols || penalty_view.ncols() != p_cols {
        return Err(py_value_error(format!(
            "penalty shape mismatch: expected {p_cols}x{p_cols}, got {}x{}",
            penalty_view.nrows(),
            penalty_view.ncols(),
        )));
    }

    let y_col: Array1<f64> = y_view.column(0).to_owned();
    let weights_owned: Array1<f64> = match weights.as_ref() {
        Some(w) => {
            let wa = w.as_array();
            if wa.len() != n_rows {
                return Err(py_value_error(format!(
                    "weights.len={} does not match N={}",
                    wa.len(),
                    n_rows,
                )));
            }
            wa.to_owned()
        }
        None => Array1::from_elem(n_rows, 1.0),
    };
    let offset_zero: Array1<f64> = Array1::zeros(n_rows);

    // Build the constraint payload. An empty A (0 rows) is treated as "no
    // constraint" — same convention used internally when no shape constraint
    // is active.
    let constraints_opt: Option<gam::pirls::LinearInequalityConstraints> =
        match (a_inequality.as_ref(), b_inequality.as_ref()) {
            (Some(a_arr), Some(b_arr)) => {
                let a_view = a_arr.as_array();
                let b_view = b_arr.as_array();
                if a_view.nrows() == 0 {
                    None
                } else {
                    if a_view.ncols() != p_cols {
                        return Err(py_value_error(format!(
                            "a_inequality has {} cols; expected {p_cols} to match X columns",
                            a_view.ncols(),
                        )));
                    }
                    if b_view.len() != a_view.nrows() {
                        return Err(py_value_error(format!(
                            "b_inequality length {} does not match a_inequality rows {}",
                            b_view.len(),
                            a_view.nrows(),
                        )));
                    }
                    Some(
                        gam::pirls::LinearInequalityConstraints::new(
                            a_view.to_owned(),
                            b_view.to_owned(),
                        )
                        .map_err(py_value_error)?,
                    )
                }
            }
            (None, None) => None,
            _ => {
                return Err(py_value_error(
                    "a_inequality and b_inequality must both be provided or both omitted"
                        .to_string(),
                ));
            }
        };

    let s_list: Vec<gam::smooth::BlockwisePenalty> = vec![gam::smooth::BlockwisePenalty::new(
        0..p_cols,
        penalty_view.to_owned(),
    )];

    let opts = gam::estimate::FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-7,
        nullspace_dims: vec![0; s_list.len()],
        linear_constraints: constraints_opt.clone(),
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };

    let heuristic_owned: Option<Vec<f64>> = init_log_lambda.map(|rho| vec![rho.exp()]);

    let x_owned = x_view.to_owned();
    let x_for_active = x_owned.clone();
    let fit = detach_estimation_result(
        py,
        "gaussian_reml_fit_with_constraints_forward",
        move || {
            let heuristic_slice = heuristic_owned.as_ref().map(|v| v.as_slice());
            gam::estimate::fit_gamwith_heuristic_lambdas(
                x_owned,
                y_col.view(),
                weights_owned.view(),
                offset_zero.view(),
                &s_list,
                heuristic_slice,
                LikelihoodSpec::new(ResponseFamily::Gaussian, InverseLink::Standard(LinkFunction::Identity)),
                &opts,
            )
        },
    )?;

    let beta = fit.beta.clone();
    let coefficients_2d = beta.clone().insert_axis(Axis(1));
    let fitted_vec: Array1<f64> = x_for_active.dot(&beta);
    let fitted_2d = fitted_vec.insert_axis(Axis(1));

    let lambdas: Array1<f64> = fit.lambdas.clone();
    let lambda_scalar = lambdas.iter().copied().next().unwrap_or(0.0);
    let log_lambda_scalar = lambda_scalar.max(1e-300).ln();

    let edf_total: f64 = fit
        .inference
        .as_ref()
        .map(|inf| inf.edf_total)
        .unwrap_or(0.0);

    // Recompute active set from final β: row i is active iff a_i·β >= b_i - tol.
    let active_indices: Vec<u64> = match constraints_opt.as_ref() {
        Some(c) if c.a.nrows() > 0 => {
            let beta_scale = beta.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
            let mut out: Vec<u64> = Vec::new();
            let ab: Array1<f64> = c.a.dot(&beta);
            for i in 0..c.a.nrows() {
                let row_scale =
                    c.a.row(i)
                        .iter()
                        .fold(0.0_f64, |m, &v| m.max(v.abs()))
                        .max(1.0);
                let tol = 1e-8 * row_scale * beta_scale.max(c.b[i].abs().max(1.0));
                if ab[i] >= c.b[i] - tol {
                    out.push(i as u64);
                }
            }
            out
        }
        _ => Vec::new(),
    };
    let active_indices_arr: Array1<u64> = Array1::from_vec(active_indices);

    let out = PyDict::new(py);
    out.set_item("coefficients", coefficients_2d.into_pyarray(py))?;
    out.set_item("fitted", fitted_2d.into_pyarray(py))?;
    out.set_item("lambda", lambda_scalar)?;
    out.set_item("log_lambda", log_lambda_scalar)?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("edf", edf_total)?;
    out.set_item("active_indices", active_indices_arr.into_pyarray(py))?;
    Ok(out.unbind())
}

/// Analytic backward (VJP) for `gaussian_reml_fit_with_constraints_forward`.
///
/// Math identity (see task spec): at the constrained cert exit with active
/// set `A_act β̂ = 0`, the envelope theorem applied to the tangent-projected
/// outer objective `V_T(ρ)` gives the same closed-form VJP as the
/// unconstrained Gaussian REML backward, with `H⁻¹` replaced by the
/// projected pseudo-inverse `P = Z (ZᵀHZ)⁻¹ Zᵀ` and `S⁺` replaced by
/// `Z (ZᵀSZ)⁺ Zᵀ` (where `Z` is the basis of `null(A_act)`).
///
/// Implementation status:
/// - **Interior cert (empty active set):** the projection `Z = I_p` is the
///   identity, so the tangent-projected VJP coincides with the unconstrained
///   closed-form Gaussian REML backward. This case delegates to
///   `gaussian_reml_multi_closed_form_backward` and produces gradients
///   identical to `gaussian_reml_fit_backward` (round-off agreement).
/// - **Active cert (non-empty active set):** STOPPED per task instructions
///   (the existing closed-form backward stack is hard-coded to the
///   unconstrained `GaussianRemlEigenCache`/`reml_hess_rho` and cannot
///   accept a tangent-wrapped operator without a substantial refactor —
///   see report). Returns `NotImplementedError`.
#[pyfunction(signature = (
    x,
    y,
    penalty,
    weights = None,
    a_inequality = None,
    b_inequality = None,
    log_lambda_at_optimum = None,
    coefficients_at_optimum = None,
    fitted_at_optimum = None,
    active_indices = None,
    grad_coefficients = None,
    grad_fitted = None,
    grad_lambda = 0.0,
    grad_log_lambda = 0.0,
    grad_reml_score = 0.0,
    grad_edf = 0.0,
))]
fn gaussian_reml_fit_with_constraints_backward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    a_inequality: Option<PyReadonlyArray2<'py, f64>>,
    b_inequality: Option<PyReadonlyArray1<'py, f64>>,
    log_lambda_at_optimum: Option<f64>,
    // `coefficients_at_optimum` is part of the documented API surface so
    // callers can pre-compute or cache it, but the analytic backward
    // derives the residual `y - X β̂` from the closed-form fit and never
    // reads this argument back.
    coefficients_at_optimum: Option<PyReadonlyArray2<'py, f64>>,
    fitted_at_optimum: Option<PyReadonlyArray2<'py, f64>>,
    active_indices: Option<PyReadonlyArray1<'py, u64>>,
    grad_coefficients: Option<PyReadonlyArray2<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_lambda: f64,
    grad_log_lambda: f64,
    grad_reml_score: f64,
    grad_edf: f64,
) -> PyResult<Py<PyDict>> {
    if let Some(b) = b_inequality.as_ref() {
        if b.as_array().iter().any(|value| value.abs() > 0.0) {
            return Err(py_value_error(
                "gaussian_reml_fit_with_constraints_backward supports only zero-bound inequality certificates".to_string(),
            ));
        }
    }
    if let Some(coefficients) = coefficients_at_optimum.as_ref() {
        let coeffs = coefficients.as_array();
        if coeffs.nrows() != x.as_array().ncols() || coeffs.ncols() != y.as_array().ncols() {
            return Err(py_value_error(format!(
                "coefficients_at_optimum shape mismatch: expected ({}, {}), got ({}, {})",
                x.as_array().ncols(),
                y.as_array().ncols(),
                coeffs.nrows(),
                coeffs.ncols()
            )));
        }
    }
    if let Some(fitted) = fitted_at_optimum.as_ref() {
        let fit = fitted.as_array();
        if fit.dim() != y.as_array().dim() {
            return Err(py_value_error(format!(
                "fitted_at_optimum shape mismatch: expected {:?}, got {:?}",
                y.as_array().dim(),
                fit.dim()
            )));
        }
    }

    // Determine whether the active set is empty (interior cert).
    let active_empty = match active_indices.as_ref() {
        Some(a) => a.as_array().len() == 0,
        None => true,
    };
    // No active constraint matrix at all is also the interior-cert case.
    let no_constraints = match a_inequality.as_ref() {
        Some(a) => a.as_array().nrows() == 0,
        None => true,
    };
    let is_interior = active_empty || no_constraints;

    if !is_interior {
        // See header doc + report: the closed-form Gaussian REML backward
        // stack is welded to the unconstrained eigen-cache. The tangent-
        // projected variant requires constructing `P = Z (ZᵀHZ)⁻¹ Zᵀ` and a
        // projected penalty pinv `Z (ZᵀSZ)⁺ Zᵀ`, plus a projected
        // `reml_hess_rho_T`. Refactoring the per-helper VJPs to accept these
        // as separate parameters instead of pulling from `cache`.
        return Err(PyNotImplementedError::new_err(
            "gaussian_reml_fit_with_constraints_backward: analytic VJP at \
             non-empty active sets (active cert exit) is not yet implemented. \
             The math identity is `H⁻¹ → Z(ZᵀHZ)⁻¹Zᵀ`, `S⁺ → Z(ZᵀSZ)⁺Zᵀ`, \
             but the closed-form Gaussian REML helpers in \
             `src/solver/gaussian_reml.rs` consume the unconstrained \
             `GaussianRemlEigenCache` directly and cannot yet accept these \
             projected operators."
                .to_string(),
        ));
    }

    // Interior cert: envelope theorem in full p-space. The constrained
    // forward converges identically to the unconstrained forward (no
    // constraint is binding), so the closed-form Gaussian REML backward
    // applied to the unconstrained problem produces the correct VJP.
    let init_lambda = log_lambda_at_optimum.map(|rho| rho.exp());

    // The constrained forward returns the smoothing parameter as `lambda`
    // and `log_lambda`. Upstream `grad_lambda` and `grad_log_lambda` both
    // pull on the same scalar; chain `grad_log_lambda` through
    // `dlog λ / dλ = 1/λ` and add to `grad_lambda`.
    let mut effective_grad_lambda = grad_lambda;
    if grad_log_lambda != 0.0 {
        let lam = init_lambda.unwrap_or(0.0);
        if lam > 0.0 {
            effective_grad_lambda += grad_log_lambda / lam;
        } else {
            // log λ undefined / unstable here. Surface a clear error rather
            // than silently zero the contribution.
            return Err(py_value_error(
                "gaussian_reml_fit_with_constraints_backward: \
                 grad_log_lambda is non-zero but log_lambda_at_optimum is \
                 missing or λ ≤ 0; cannot chain dlog λ/dλ = 1/λ."
                    .to_string(),
            ));
        }
    }

    let x_values = x.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let grad_coefficients_values = grad_coefficients.as_ref().map(|g| g.as_array().to_owned());
    let grad_fitted_values = grad_fitted.as_ref().map(|g| g.as_array().to_owned());
    let backward = detach_py_result(
        py,
        "gaussian_reml_fit_with_constraints_backward",
        move || {
            gaussian_reml_multi_closed_form_backward(
                x_values.view(),
                y_values.view(),
                penalty_values.view(),
                weight_values.as_ref().map(|w| w.view()),
                init_lambda,
                effective_grad_lambda,
                grad_coefficients_values.as_ref().map(|g| g.view()),
                grad_fitted_values.as_ref().map(|g| g.view()),
                grad_reml_score,
                grad_edf,
            )
            .map_err(|err| err.to_string())
        },
    )?;

    let out = PyDict::new(py);
    out.set_item("grad_x", backward.grad_x.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", backward.grad_penalty.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction(signature = (
    x,
    y,
    row_offsets,
    penalty,
    weights = None,
    init_lambda = None,
    by = None,
    by_start_col = 0
))]
fn gaussian_reml_fit_batched<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    row_offsets: PyReadonlyArray1<'py, usize>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let x_values = x.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let row_offset_values = row_offsets.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let result = detach_py_result(py, "gaussian_reml_fit_batched", move || {
        let gated_x;
        let fit_x = if let Some(by_arr) = by_values.as_ref() {
            gated_x = apply_by_gate(x_values.view(), by_arr.view(), by_start_col)?;
            gated_x.view()
        } else {
            x_values.view()
        };
        gaussian_reml_fit_batched_impl(
            fit_x,
            y_values.view(),
            row_offset_values.view(),
            penalty_values.view(),
            weight_values.as_ref().map(|w| w.view()),
            init_lambda,
        )
    })?;
    let out = PyDict::new(py);
    set_batched_gaussian_reml_dict_items(py, &out, result)?;
    Ok(out.unbind())
}

fn set_batched_gaussian_reml_dict_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    result: BatchedGaussianRemlResult,
) -> PyResult<()> {
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
    out.set_item(
        "cache_penalty_eigenvalues",
        result.cache_penalty_eigenvalues.into_pyarray(py),
    )?;
    out.set_item(
        "cache_eigenvectors",
        result.cache_eigenvectors.into_pyarray(py),
    )?;
    out.set_item(
        "cache_coefficient_basis",
        result.cache_coefficient_basis.into_pyarray(py),
    )?;
    out.set_item(
        "cache_xtwx_fingerprints",
        result.cache_xtwx_fingerprints.into_pyarray(py),
    )?;
    out.set_item(
        "cache_penalty_fingerprints",
        result.cache_penalty_fingerprints.into_pyarray(py),
    )?;
    out.set_item(
        "cache_logdet_xtwx",
        result.cache_logdet_xtwx.into_pyarray(py),
    )?;
    out.set_item(
        "cache_logdet_penalty_positive",
        result.cache_logdet_penalty_positive.into_pyarray(py),
    )?;
    out.set_item(
        "cache_penalty_ranks",
        result.cache_penalty_ranks.into_pyarray(py),
    )?;
    out.set_item("cache_nullities", result.cache_nullities.into_pyarray(py))?;
    Ok(())
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
    grad_edf = None,
    forward_state = None,
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
    grad_edf: Option<PyReadonlyArray1<'py, f64>>,
    forward_state: Option<&Bound<'py, PyDict>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let forward_fits = forward_state
        .map(|state| batched_gaussian_reml_fits_from_pydict(state, row_offsets.as_array()))
        .transpose()
        .map_err(py_value_error)?;
    let x_values = x.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let row_offset_values = row_offsets.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let grad_lambda_values = grad_lambda.as_ref().map(|g| g.as_array().to_owned());
    let grad_coefficients_values = grad_coefficients.as_ref().map(|g| g.as_array().to_owned());
    let grad_fitted_values = grad_fitted.as_ref().map(|g| g.as_array().to_owned());
    let grad_reml_score_values = grad_reml_score.as_ref().map(|g| g.as_array().to_owned());
    let grad_edf_values = grad_edf.as_ref().map(|g| g.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let (statuses, grad_x, grad_by, grad_y, grad_penalty, grad_weights) =
        detach_py_result(py, "gaussian_reml_fit_batched_backward", move || {
            let gated_x;
            let fit_x = if let Some(by_arr) = by_values.as_ref() {
                gated_x = apply_by_gate(x_values.view(), by_arr.view(), by_start_col)?;
                gated_x.view()
            } else {
                x_values.view()
            };
            let backward = gaussian_reml_fit_batched_backward_impl(
                fit_x,
                y_values.view(),
                row_offset_values.view(),
                penalty_values.view(),
                weight_values.as_ref().map(|w| w.view()),
                init_lambda,
                grad_lambda_values.as_ref().map(|g| g.view()),
                grad_coefficients_values.as_ref().map(|g| g.view()),
                grad_fitted_values.as_ref().map(|g| g.view()),
                grad_reml_score_values.as_ref().map(|g| g.view()),
                grad_edf_values.as_ref().map(|g| g.view()),
                forward_fits.as_deref(),
            )?;
            let grad_x_raw = backward.grad_x;
            let (grad_x, grad_by) = if let Some(by_arr) = by_values.as_ref() {
                let (grad_x, grad_by) = apply_by_gate_backward(
                    x_values.view(),
                    by_arr.view(),
                    by_start_col,
                    grad_x_raw.view(),
                )?;
                (grad_x, Some(grad_by))
            } else {
                (grad_x_raw, None)
            };
            Ok((
                backward.statuses,
                grad_x,
                grad_by,
                backward.grad_y,
                backward.grad_penalty,
                backward.grad_weights,
            ))
        })?;

    let out = PyDict::new(py);
    out.set_item("status", statuses)?;
    out.set_item("grad_x", grad_x.into_pyarray(py))?;
    out.set_item("grad_y", grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", grad_penalty.into_pyarray(py))?;
    out.set_item("grad_weights", grad_weights.into_pyarray(py))?;
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
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let knot_or_center_values = knots_or_centers.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let n_rows = t_values.len();
    let n_outputs = y_values.ncols();
    let n_coefficients = penalty_values.nrows();
    let result = detach_py_result(py, "gaussian_reml_fit_positions", move || {
        let x = position_basis_design(
            t_values.view(),
            knot_or_center_values.view(),
            &basis_kind,
            basis_order,
            periodic,
            period,
        )?;
        let gated_x;
        let fit_x = if let Some(by_arr) = by_values.as_ref() {
            gated_x = apply_by_gate(x.view(), by_arr.view(), by_start_col)?;
            gated_x.view()
        } else {
            x.view()
        };
        match gaussian_reml_multi_closed_form_with_cache(
            fit_x,
            y_values.view(),
            penalty_values.view(),
            weight_values.as_ref().map(|w| w.view()),
            init_lambda,
            None,
        ) {
            Ok(fit) => Ok(Some(fit)),
            Err(EstimationError::ModelIsIllConditioned { .. }) => Ok(None),
            Err(err) => Err(err.to_string()),
        }
    })?;
    let out = PyDict::new(py);
    match result {
        Some(fit) => set_ok_gaussian_reml_items(py, &out, fit)?,
        None => {
            set_degenerate_gaussian_reml_items(py, &out, n_rows, n_outputs, n_coefficients)?;
        }
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
    grad_edf = 0.0,
    forward_state = None,
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
    grad_edf: f64,
    forward_state: Option<&Bound<'py, PyDict>>,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let forward_fit = forward_state
        .map(gaussian_reml_fit_state_from_pydict)
        .transpose()
        .map_err(py_value_error)?;
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let knot_or_center_values = knots_or_centers.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let grad_coefficients_values = grad_coefficients.as_ref().map(|g| g.as_array().to_owned());
    let grad_fitted_values = grad_fitted.as_ref().map(|g| g.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let backward = detach_py_result(py, "gaussian_reml_fit_positions_backward", move || {
        gaussian_reml_fit_positions_backward_impl(
            t_values.view(),
            y_values.view(),
            knot_or_center_values.view(),
            &basis_kind,
            basis_order,
            periodic,
            period,
            penalty_values.view(),
            weight_values.as_ref().map(|w| w.view()),
            init_lambda,
            grad_lambda,
            grad_coefficients_values.as_ref().map(|g| g.view()),
            grad_fitted_values.as_ref().map(|g| g.view()),
            grad_reml_score,
            grad_edf,
            by_values.as_ref().map(|b| b.view()),
            by_start_col,
            forward_fit.as_ref(),
        )
    })?;

    let out = PyDict::new(py);
    out.set_item("grad_t", backward.grad_t.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", backward.grad_penalty.into_pyarray(py))?;
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
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let row_offset_values = row_offsets.as_array().to_owned();
    let knot_or_center_values = knots_or_centers.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let result = detach_py_result(py, "gaussian_reml_fit_positions_batched", move || {
        gaussian_reml_fit_positions_batched_impl(
            t_values.view(),
            y_values.view(),
            row_offset_values.view(),
            knot_or_center_values.view(),
            &basis_kind,
            basis_order,
            periodic,
            period,
            penalty_values.view(),
            weight_values.as_ref().map(|w| w.view()),
            init_lambda,
            by_values.as_ref().map(|b| b.view()),
            by_start_col,
        )
    })?;
    let out = PyDict::new(py);
    set_batched_gaussian_reml_dict_items(py, &out, result)?;
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
    grad_edf = None,
    forward_state = None,
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
    grad_edf: Option<PyReadonlyArray1<'py, f64>>,
    forward_state: Option<&Bound<'py, PyDict>>,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let forward_fits = forward_state
        .map(|state| batched_gaussian_reml_fits_from_pydict(state, row_offsets.as_array()))
        .transpose()
        .map_err(py_value_error)?;
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let row_offset_values = row_offsets.as_array().to_owned();
    let knot_or_center_values = knots_or_centers.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let grad_lambda_values = grad_lambda.as_ref().map(|g| g.as_array().to_owned());
    let grad_coefficients_values = grad_coefficients.as_ref().map(|g| g.as_array().to_owned());
    let grad_fitted_values = grad_fitted.as_ref().map(|g| g.as_array().to_owned());
    let grad_reml_score_values = grad_reml_score.as_ref().map(|g| g.as_array().to_owned());
    let grad_edf_values = grad_edf.as_ref().map(|g| g.as_array().to_owned());
    let by_values = by.as_ref().map(|b_arr| b_arr.as_array().to_owned());
    let backward = detach_py_result(
        py,
        "gaussian_reml_fit_positions_batched_backward",
        move || {
            gaussian_reml_fit_positions_batched_backward_impl(
                t_values.view(),
                y_values.view(),
                row_offset_values.view(),
                knot_or_center_values.view(),
                &basis_kind,
                basis_order,
                periodic,
                period,
                penalty_values.view(),
                weight_values.as_ref().map(|w| w.view()),
                init_lambda,
                grad_lambda_values.as_ref().map(|g| g.view()),
                grad_coefficients_values.as_ref().map(|g| g.view()),
                grad_fitted_values.as_ref().map(|g| g.view()),
                grad_reml_score_values.as_ref().map(|g| g.view()),
                grad_edf_values.as_ref().map(|g| g.view()),
                by_values.as_ref().map(|b_arr| b_arr.view()),
                by_start_col,
                forward_fits.as_deref(),
            )
        },
    )?;

    let out = PyDict::new(py);
    out.set_item("status", backward.statuses)?;
    out.set_item("grad_t", backward.grad_t.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", backward.grad_penalty.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    if let Some(grad_by) = backward.grad_by {
        out.set_item("grad_by", grad_by.into_pyarray(py))?;
    } else {
        out.set_item("grad_by", py.None())?;
    }
    Ok(out.unbind())
}

// ---------------------------------------------------------------------------
// LatentCoord — N-D generalization of `gaussian_reml_fit_positions`
// ---------------------------------------------------------------------------
//
// See `proposals/latent_coord.md` and `src/terms/latent_coord.rs`.
//
// The 1-D position path constructs Φ(t) on a Duchon/B-spline basis with
// t ∈ ℝ^N, fits the Gaussian REML inner problem against Y, and (in the
// backward call) contracts ∂L/∂Φ with the basis derivative ∂Φ/∂t to
// produce grad_t. The latent path is the same construction lifted to
// t ∈ ℝ^{N × d}:
//
//   * design Φ_{n,k} = K(t_n, c_k) is built by `build_duchon_basis`
//     with N-D `data` and `centers` (an existing entry point);
//   * radial first derivative `φ'(r_{nk})` is computed by the new
//     `duchon_radial_first_derivative_nd` basis helper;
//   * `∂Φ/∂t` is assembled at the call site via the
//     per-basis `*_first_derivative_nd` helpers in `gam::terms::basis`;
//   * `grad_t` is the contraction
//     `gam::terms::input_loc_derivatives::contract_input_loc_gradient(grad_phi, jet)`.
//
// Identifiability modes (`LatentIdMode::AuxPrior`, `DimSelection`) are
// folded into the inner Gaussian REML call via virtual-row augmentation:
// adding `√μ` rows that pull `t` toward a target (or zero, for ARD)
// turns the gauge-flat valley into a strict minimum without modifying
// the inner solver. This is exactly the iVAE / ARD recasting from the
// proposal §4(c), §4(d).

fn build_latent_duchon_design(
    t_flat: ArrayView1<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if t_flat.len() != n_obs * latent_dim {
        return Err(format!(
            "latent t length {} != n_obs * latent_dim = {}",
            t_flat.len(),
            n_obs * latent_dim
        ));
    }
    if centers.ncols() != latent_dim {
        return Err(format!(
            "centers must have {latent_dim} columns to match latent_dim; got {}",
            centers.ncols()
        ));
    }
    if m == 0 {
        return Err("LatentCoord Duchon m must be at least 1".into());
    }
    // Materialize t as a (n_obs, latent_dim) matrix.
    let mut t_mat = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            t_mat[[n, a]] = t_flat[n * latent_dim + a];
        }
    }
    let center_matrix = centers.to_owned();
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(center_matrix.clone()),
        length_scale: None,
        power: 0.0,
        nullspace_order: duchon_nullspace_from_m(m),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    let built = build_duchon_basis(t_mat.view(), &spec)
        .map_err(|err| format!("failed to evaluate N-D Duchon basis for LatentCoord: {err}"))?;
    let design = built
        .design
        .try_to_dense_by_chunks("latent_duchon_design")
        .map_err(|err| format!("failed to evaluate N-D Duchon basis for LatentCoord: {err}"))?;
    Ok((design, t_mat))
}

fn t_matrix_from_flat(
    t_flat: ArrayView1<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
) -> Result<Array2<f64>, String> {
    if t_flat.len() != n_obs * latent_dim {
        return Err(format!(
            "latent t length {} != n_obs * latent_dim = {}",
            t_flat.len(),
            n_obs * latent_dim
        ));
    }
    let mut t_mat = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            t_mat[[n, a]] = t_flat[n * latent_dim + a];
        }
    }
    Ok(t_mat)
}

fn split_tensor_knots_owned(
    knots_concat: ArrayView1<'_, f64>,
    knot_offsets: &[usize],
    n_axes: usize,
) -> Result<Vec<Array1<f64>>, String> {
    if knot_offsets.len() != n_axes + 1 {
        return Err(format!(
            "tensor B-spline knot_offsets must have length n_axes + 1 = {}, got {}",
            n_axes + 1,
            knot_offsets.len()
        ));
    }
    let mut per_axis = Vec::with_capacity(n_axes);
    for axis in 0..n_axes {
        let lo = knot_offsets[axis];
        let hi = knot_offsets[axis + 1];
        if lo > hi || hi > knots_concat.len() {
            return Err(format!(
                "tensor B-spline knot_offsets axis {axis} out of range \
                 (lo={lo}, hi={hi}, total={})",
                knots_concat.len()
            ));
        }
        per_axis.push(knots_concat.slice(s![lo..hi]).to_owned());
    }
    Ok(per_axis)
}

fn build_latent_tensor_bspline_design(
    t_flat: ArrayView1<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    knots_concat: ArrayView1<'_, f64>,
    knot_offsets: &[usize],
    degrees: &[usize],
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if degrees.len() != latent_dim {
        return Err(format!(
            "tensor B-spline degrees length {} must equal latent_dim {}",
            degrees.len(),
            latent_dim
        ));
    }
    let t_mat = t_matrix_from_flat(t_flat, n_obs, latent_dim)?;
    let knots_per_axis = split_tensor_knots_owned(knots_concat, knot_offsets, latent_dim)?;
    let knot_views = knots_per_axis
        .iter()
        .map(|knots| knots.view())
        .collect::<Vec<_>>();
    let mut k_per_axis = Vec::<usize>::with_capacity(latent_dim);
    let mut total_cols = 1usize;
    for axis in 0..latent_dim {
        let k = knot_views[axis]
            .len()
            .checked_sub(degrees[axis] + 1)
            .ok_or_else(|| {
                format!(
                    "tensor B-spline axis {axis} knot vector too short for degree {}",
                    degrees[axis]
                )
            })?;
        k_per_axis.push(k);
        total_cols = total_cols
            .checked_mul(k)
            .ok_or_else(|| "tensor B-spline basis size overflow".to_string())?;
    }

    let mut design = Array2::<f64>::zeros((n_obs, total_cols));
    let mut values_per_axis: Vec<Vec<f64>> = k_per_axis.iter().map(|&k| vec![0.0; k]).collect();
    let mut scratch: Vec<SplineScratch> = (0..latent_dim)
        .map(|axis| SplineScratch::new(degrees[axis]))
        .collect();
    let mut idx = vec![0usize; latent_dim];
    for n in 0..n_obs {
        for axis in 0..latent_dim {
            evaluate_bspline_basis_scalar(
                t_mat[[n, axis]],
                knot_views[axis],
                degrees[axis],
                &mut values_per_axis[axis],
                &mut scratch[axis],
            )
            .map_err(|err| {
                format!("failed to evaluate tensor B-spline latent axis {axis}: {err}")
            })?;
        }
        for col in 0..total_cols {
            let mut rem = col;
            for axis in (0..latent_dim).rev() {
                idx[axis] = rem % k_per_axis[axis];
                rem /= k_per_axis[axis];
            }
            let mut prod = 1.0_f64;
            for axis in 0..latent_dim {
                prod *= values_per_axis[axis][idx[axis]];
            }
            design[[n, col]] = prod;
        }
    }
    Ok((design, t_mat))
}

fn latent_periodic_range_from_centers(centers: ArrayView2<'_, f64>) -> Result<(f64, f64), String> {
    if centers.ncols() != 1 || centers.nrows() == 0 {
        return Err("periodic B-spline latent design requires one-column centers".to_string());
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &value in centers.column(0).iter() {
        lo = lo.min(value);
        hi = hi.max(value);
    }
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return Err("periodic B-spline centers must define a finite range".to_string());
    }
    Ok((lo, hi))
}

fn project_latent_jet_columns(
    raw_jet: &Array3<f64>,
    transform: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, String> {
    let n_rows = raw_jet.shape()[0];
    let raw_cols = raw_jet.shape()[1];
    let latent_dim = raw_jet.shape()[2];
    if transform.nrows() != raw_cols {
        return Err(format!(
            "latent jet transform row mismatch: jet has {raw_cols} columns, transform has {} rows",
            transform.nrows()
        ));
    }
    let mut out = Array3::<f64>::zeros((n_rows, transform.ncols(), latent_dim));
    for n in 0..n_rows {
        for j in 0..transform.ncols() {
            for k in 0..raw_cols {
                let z = transform[[k, j]];
                if z == 0.0 {
                    continue;
                }
                for a in 0..latent_dim {
                    out[[n, j, a]] += raw_jet[[n, k, a]] * z;
                }
            }
        }
    }
    Ok(out)
}

fn build_latent_forward_design(
    basis_kind: &str,
    t_flat: ArrayView1<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
) -> Result<(Array2<f64>, Array2<f64>, Array3<f64>), String> {
    let basis_kind = latent_basis_kind(basis_kind)?;
    let (design, t_mat) = match basis_kind {
        "duchon" => build_latent_duchon_design(t_flat, n_obs, latent_dim, centers, m)?,
        "matern" => {
            if centers.ncols() != latent_dim {
                return Err(format!(
                    "Matérn latent centers must have {latent_dim} columns; got {}",
                    centers.ncols()
                ));
            }
            let t_mat = t_matrix_from_flat(t_flat, n_obs, latent_dim)?;
            let spec = MaternBasisSpec {
                center_strategy: CenterStrategy::UserProvided(centers.to_owned()),
                length_scale: 1.0,
                nu: MaternNu::ThreeHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::None,
                aniso_log_scales: None,
                periodic: None,
            };
            let built = build_matern_basis(t_mat.view(), &spec)
                .map_err(|err| format!("failed to evaluate Matérn latent basis: {err}"))?;
            let design = built
                .design
                .try_to_dense_by_chunks("latent_matern_design")
                .map_err(|err| format!("failed to evaluate Matérn latent basis: {err}"))?;
            (design, t_mat)
        }
        "sphere" => {
            if centers.ncols() != latent_dim {
                return Err(format!(
                    "sphere latent centers must have {latent_dim} columns; got {}",
                    centers.ncols()
                ));
            }
            let t_mat = t_matrix_from_flat(t_flat, n_obs, latent_dim)?;
            let spec = SphericalSplineBasisSpec {
                center_strategy: CenterStrategy::UserProvided(centers.to_owned()),
                penalty_order: m,
                double_penalty: false,
                radians: true,
                method: SphereMethod::Wahba,
                max_degree: None,
                wahba_kernel: SphereWahbaKernel::Sobolev,
            };
            let built = build_spherical_spline_basis(t_mat.view(), &spec)
                .map_err(|err| format!("failed to evaluate sphere latent basis: {err}"))?;
            let constraint_transform = match &built.metadata {
                gam::basis::BasisMetadata::Sphere {
                    constraint_transform,
                    ..
                } => constraint_transform.clone(),
                _ => None,
            };
            let design = built
                .design
                .try_to_dense_by_chunks("latent_sphere_design")
                .map_err(|err| format!("failed to evaluate sphere latent basis: {err}"))?;
            let raw_jet = latent_input_location_jet(
                basis_kind,
                t_mat.view(),
                centers,
                m,
                tensor_knots_concat,
                tensor_knot_offsets,
                tensor_degrees,
            )?;
            let jet = match constraint_transform {
                Some(z) => project_latent_jet_columns(&raw_jet, z.view())?,
                _ => raw_jet,
            };
            if jet.shape()[1] != design.ncols() {
                return Err(format!(
                    "sphere latent design/jet column mismatch: design has {}, jet has {}",
                    design.ncols(),
                    jet.shape()[1]
                ));
            }
            return Ok((design, t_mat, jet));
        }
        "bspline_tensor" => {
            let knots = tensor_knots_concat
                .as_ref()
                .ok_or_else(|| "tensor B-spline latent design requires knots_concat".to_string())?
                .clone();
            let offsets = tensor_knot_offsets
                .ok_or_else(|| "tensor B-spline latent design requires knot_offsets".to_string())?;
            let degrees = tensor_degrees
                .ok_or_else(|| "tensor B-spline latent design requires degrees".to_string())?;
            build_latent_tensor_bspline_design(t_flat, n_obs, latent_dim, knots, offsets, degrees)?
        }
        "periodic_bspline" => {
            if latent_dim != 1 {
                return Err(format!(
                    "periodic B-spline latent design requires latent_dim 1; got {latent_dim}"
                ));
            }
            let t_mat = t_matrix_from_flat(t_flat, n_obs, latent_dim)?;
            let range = latent_periodic_range_from_centers(centers)?;
            let design =
                periodic_bspline_basis_dense_via_spec(t_mat.column(0), range, m, centers.nrows())?;
            (design, t_mat)
        }
        other => {
            return Err(format!(
                "gaussian_reml_fit_latent does not support latent basis_kind {other:?}"
            ));
        }
    };
    let jet = latent_input_location_jet(
        basis_kind,
        t_mat.view(),
        centers,
        m,
        tensor_knots_concat,
        tensor_knot_offsets,
        tensor_degrees,
    )?;
    if jet.shape()[1] != design.ncols() {
        return Err(format!(
            "latent design/jet column mismatch for {basis_kind:?}: design has {}, jet has {}",
            design.ncols(),
            jet.shape()[1]
        ));
    }
    Ok((design, t_mat, jet))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SigmaEffMode {
    Profiled,
    Fixed,
}

impl SigmaEffMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "profiled" | "profile" | "reml" => Ok(Self::Profiled),
            "fixed" | "sigma" | "sigma2" => Ok(Self::Fixed),
            other => Err(format!(
                "sigma_eff_mode must be 'profiled' or 'fixed'; got {other:?}"
            )),
        }
    }
}

fn latent_basis_kind(value: &str) -> Result<&'static str, String> {
    match value.to_ascii_lowercase().replace(['_', '-'], "").as_str() {
        "duchon" | "duchonspline" => Ok("duchon"),
        // Dispatch hooks for the in-flight non-Duchon derivative helpers.
        // The call sites below are intentionally shaped around
        // `InputLocationDerivative::{Radial, Jet}` so Matérn can plug into
        // the radial path and sphere / tensor / periodic bases can plug into
        // the pre-computed-jet path without changing the contraction code.
        "matern" | "maternradial" => Ok("matern"),
        "sphere" | "spherical" => Ok("sphere"),
        "bsplinetensor" | "tensorbspline" => Ok("bspline_tensor"),
        "periodicbspline" | "periodicspline" => Ok("periodic_bspline"),
        other => Err(format!("unsupported latent basis_kind {other:?}")),
    }
}

fn radial_input_location_jet(
    t_mat: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    phi_r: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, String> {
    if phi_r.dim() != (t_mat.nrows(), centers.nrows()) {
        return Err(format!(
            "radial derivative shape {:?} does not match t/centers ({}, {})",
            phi_r.dim(),
            t_mat.nrows(),
            centers.nrows()
        ));
    }
    if t_mat.ncols() != centers.ncols() {
        return Err(format!(
            "radial derivative dimension mismatch: t has {} cols, centers has {}",
            t_mat.ncols(),
            centers.ncols()
        ));
    }
    let mut out = Array3::<f64>::zeros((t_mat.nrows(), centers.nrows(), t_mat.ncols()));
    for n in 0..t_mat.nrows() {
        for k in 0..centers.nrows() {
            let mut r2 = 0.0;
            for a in 0..t_mat.ncols() {
                let delta = t_mat[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            if r <= 1.0e-12 {
                continue;
            }
            let scale = phi_r[[n, k]] / r;
            for a in 0..t_mat.ncols() {
                out[[n, k, a]] = scale * (t_mat[[n, a]] - centers[[k, a]]);
            }
        }
    }
    Ok(out)
}

fn latent_input_location_jet(
    basis_kind: &str,
    t_mat: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    m: usize,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
) -> Result<Array3<f64>, String> {
    match latent_basis_kind(basis_kind)? {
        "duchon" => {
            let nullspace_order = duchon_nullspace_from_m(m);
            let phi_r = duchon_radial_first_derivative_nd(t_mat, centers, None, nullspace_order)
                .map_err(|err| err.to_string())?;
            let radial_jet = radial_input_location_jet(t_mat, centers, phi_r.view())?;
            let poly_jet = duchon_polynomial_first_derivative_nd(t_mat, nullspace_order);
            let n_rows = radial_jet.shape()[0];
            let n_radial = radial_jet.shape()[1];
            let n_poly = poly_jet.shape()[1];
            let dim = radial_jet.shape()[2];
            if poly_jet.shape()[0] != n_rows || poly_jet.shape()[2] != dim {
                return Err(format!(
                    "Duchon polynomial derivative shape mismatch: radial jet is \
                     {}x{}x{}, polynomial jet is {}x{}x{}",
                    n_rows,
                    n_radial,
                    dim,
                    poly_jet.shape()[0],
                    n_poly,
                    poly_jet.shape()[2],
                ));
            }
            let mut jet = Array3::<f64>::zeros((n_rows, n_radial + n_poly, dim));
            for n in 0..n_rows {
                for k in 0..n_radial {
                    for a in 0..dim {
                        jet[[n, k, a]] = radial_jet[[n, k, a]];
                    }
                }
                for k in 0..n_poly {
                    for a in 0..dim {
                        jet[[n, n_radial + k, a]] = poly_jet[[n, k, a]];
                    }
                }
            }
            Ok(jet)
        }
        "matern" => {
            // Fixes audit-revised claim that non-Duchon latent input-location
            // derivatives must use the closed-form helper instead of stubbing.
            let phi_r =
                matern_radial_first_derivative_nd(t_mat, centers, 1.0, MaternNu::ThreeHalves)
                    .map_err(|err| err.to_string())?;
            radial_input_location_jet(t_mat, centers, phi_r.view())
        }
        "sphere" => {
            // Fixes audit-revised claim that sphere latent derivatives are
            // analytic jets, not unsupported hooks.
            let jet = sphere_first_derivative_nd(t_mat, centers, m, true)
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        "bspline_tensor" => {
            let knots = tensor_knots_concat.ok_or_else(|| {
                "tensor B-spline latent derivative requires knots_concat".to_string()
            })?;
            let offsets = tensor_knot_offsets.ok_or_else(|| {
                "tensor B-spline latent derivative requires knot_offsets".to_string()
            })?;
            let degrees = tensor_degrees
                .ok_or_else(|| "tensor B-spline latent derivative requires degrees".to_string())?;
            let per_axis = split_tensor_knots_owned(knots, offsets, t_mat.ncols())?;
            let per_axis_views = per_axis
                .iter()
                .map(|axis_knots| axis_knots.view())
                .collect::<Vec<_>>();
            let jet = bspline_tensor_first_derivative(t_mat, &per_axis_views, degrees)
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        "periodic_bspline" => {
            // Fixes audit-revised claim that periodic latent derivatives are
            // analytic jets. The latent pyffi path carries only centers today,
            // so infer the period from the first center column.
            if centers.ncols() != 1 || centers.nrows() == 0 {
                return Err(
                    "periodic B-spline latent derivative requires one-column centers".to_string(),
                );
            }
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &value in centers.column(0).iter() {
                lo = lo.min(value);
                hi = hi.max(value);
            }
            if !(lo.is_finite() && hi.is_finite() && hi > lo) {
                return Err("periodic B-spline centers must define a finite range".to_string());
            }
            let jet = periodic_bspline_first_derivative_nd(t_mat, (lo, hi), m, centers.nrows())
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        other => Err(format!(
            "latent_basis_kind returned an unknown normalized kind: {other}"
        )),
    }
}

fn gaussian_reml_weight_vector_local(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, String> {
    match weights {
        Some(w) => {
            if w.len() != n_obs {
                return Err(format!(
                    "Gaussian REML weights length mismatch: expected {n_obs}, got {}",
                    w.len()
                ));
            }
            if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err("Gaussian REML weights must be finite and non-negative".to_string());
            }
            Ok(w.to_owned())
        }
        None => Ok(Array1::ones(n_obs)),
    }
}

fn latent_scalar_weights_with_fisher(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
    fisher_w: Option<ArrayView3<'_, f64>>,
) -> Result<Option<Array1<f64>>, String> {
    let Some(fw) = fisher_w else {
        return Ok(weights.map(|w| w.to_owned()));
    };
    if fw.shape() != [n_obs, 1, 1] {
        return Err(format!(
            "fisher_w currently accepts scalar blocks of shape ({n_obs}, 1, 1) on this latent entry point; got {:?}",
            fw.shape()
        ));
    }
    let mut out = match weights {
        Some(w) => gaussian_reml_weight_vector_local(n_obs, Some(w))?,
        None => Array1::ones(n_obs),
    };
    for n in 0..n_obs {
        let v = fw[[n, 0, 0]];
        if !(v.is_finite() && v >= 0.0) {
            return Err(format!(
                "fisher_w[{n},0,0] must be finite and non-negative; got {v}"
            ));
        }
        out[n] *= v;
    }
    Ok(Some(out))
}

fn latent_row_weights(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, String> {
    match weights {
        Some(w) => gaussian_reml_weight_vector_local(n_obs, Some(w)),
        None => Ok(Array1::ones(n_obs)),
    }
}

fn validate_dense_fisher_w(
    n_obs: usize,
    n_outputs: usize,
    fisher_w: ArrayView3<'_, f64>,
) -> Result<(), String> {
    if fisher_w.shape() != [n_obs, n_outputs, n_outputs] {
        return Err(format!(
            "fisher_w dense blocks must have shape ({n_obs}, {n_outputs}, {n_outputs}); got {:?}",
            fisher_w.shape()
        ));
    }
    for n in 0..n_obs {
        for a in 0..n_outputs {
            for b in 0..n_outputs {
                let v = fisher_w[[n, a, b]];
                if !v.is_finite() {
                    return Err(format!("fisher_w[{n},{a},{b}] must be finite; got {v}"));
                }
            }
            if fisher_w[[n, a, a]] < 0.0 {
                return Err(format!(
                    "fisher_w[{n},{a},{a}] must be non-negative; got {}",
                    fisher_w[[n, a, a]]
                ));
            }
        }
    }
    Ok(())
}

fn add_block_diagonal_penalty(
    hessian: &mut Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    lambda: f64,
    n_outputs: usize,
) -> Result<(), String> {
    let k = penalty.ncols();
    if penalty.nrows() != k {
        return Err(format!(
            "penalty must be square for dense Fisher fit; got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if hessian.dim() != (k * n_outputs, k * n_outputs) {
        return Err("dense Fisher Hessian shape mismatch while adding penalty".to_string());
    }
    for output in 0..n_outputs {
        let offset = output * k;
        for row in 0..k {
            for col in 0..k {
                let s_sym = 0.5 * (penalty[[row, col]] + penalty[[col, row]]);
                hessian[[offset + row, offset + col]] += lambda * s_sym;
            }
        }
    }
    Ok(())
}

fn solve_dense_block_system(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    context: &str,
) -> Result<Array1<f64>, String> {
    let mut rhs2 = Array2::<f64>::zeros((rhs.len(), 1));
    for i in 0..rhs.len() {
        rhs2[[i, 0]] = rhs[i];
    }
    let factor = factorize_symmetricwith_fallback(
        gam::faer_ndarray::FaerArrayView::new(hessian).as_ref(),
        Side::Lower,
    )
    .map_err(|err| format!("{context} factorization failed: {err}"))?;
    {
        let mut rhs_view = array2_to_matmut(&mut rhs2);
        factor.solve_in_place(rhs_view.as_mut());
    }
    let mut out = Array1::<f64>::zeros(rhs.len());
    for i in 0..rhs.len() {
        out[i] = rhs2[[i, 0]];
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(format!("{context} solve produced non-finite coefficients"));
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct LatentAuxStrengthState {
    log_mu: f64,
    mu: f64,
    auto: bool,
}

struct LatentAuxPriorStats {
    targets: Array2<f64>,
    residual_sq: f64,
    strength: LatentAuxStrengthState,
    score: f64,
}

fn latent_aux_prior_stats(
    t_mat: ArrayView2<'_, f64>,
    u_view: ArrayView2<'_, f64>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
) -> Result<LatentAuxPriorStats, String> {
    let targets = aux_prior_targets(t_mat, u_view, aux_family)?;
    let n_obs = t_mat.nrows();
    let latent_dim = t_mat.ncols();
    let mut residual_sq = 0.0_f64;
    for n in 0..n_obs {
        for a in 0..latent_dim {
            let diff = t_mat[[n, a]] - targets[[n, a]];
            residual_sq += diff * diff;
        }
    }
    if !residual_sq.is_finite() {
        return Err("auxiliary prior residual norm must be finite".to_string());
    }
    // The pyffi latent entry points receive `t` as the current outer
    // coordinate. When Python passes None/"auto", the log_mu coordinate has
    // a closed-form REML optimum for this fixed t because only the normalized
    // auxiliary prior depends on it.
    let (log_mu, mu, auto) = match aux_strength {
        Some(mu) => {
            if !(mu.is_finite() && mu > 0.0) {
                return Err(format!(
                    "aux_strength must be finite and positive; got {mu}"
                ));
            }
            (mu.ln(), mu, false)
        }
        None => {
            if residual_sq <= 0.0 {
                return Err(
                    "aux_strength='auto' has no finite REML optimum when the auxiliary residual is zero"
                        .to_string(),
                );
            }
            let mu = (n_obs as f64) / residual_sq;
            if !(mu.is_finite() && mu > 0.0) {
                return Err(format!(
                    "auto aux_strength selected a non-finite precision: {mu}"
                ));
            }
            (mu.ln(), mu, true)
        }
    };
    let score = 0.5 * mu * residual_sq - 0.5 * (n_obs as f64) * log_mu;
    Ok(LatentAuxPriorStats {
        targets,
        residual_sq,
        strength: LatentAuxStrengthState { log_mu, mu, auto },
        score,
    })
}

fn set_aux_strength_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    state: Option<LatentAuxStrengthState>,
) -> PyResult<()> {
    if let Some(state) = state {
        out.set_item("aux_strength", state.mu)?;
        out.set_item("aux_log_strength", state.log_mu)?;
        out.set_item("log_mu", state.log_mu)?;
        out.set_item(
            "aux_strength_mode",
            if state.auto { "auto" } else { "fixed" },
        )?;
    } else {
        out.set_item("aux_strength", py.None())?;
        out.set_item("aux_log_strength", py.None())?;
        out.set_item("log_mu", py.None())?;
        out.set_item("aux_strength_mode", py.None())?;
    }
    Ok(())
}

fn latent_prior_score_and_aux_state_for_t(
    t_mat: ArrayView2<'_, f64>,
    aux_u: Option<ArrayView2<'_, f64>>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
    dim_selection_precision: Option<ArrayView1<'_, f64>>,
) -> Result<(f64, Option<LatentAuxStrengthState>), String> {
    let n_obs = t_mat.nrows();
    let latent_dim = t_mat.ncols();
    let mut latent_prior_score = 0.0_f64;
    let mut aux_strength_state = None;
    if let Some(u_view) = aux_u {
        let stats = latent_aux_prior_stats(t_mat, u_view, aux_family, aux_strength)?;
        latent_prior_score += stats.score;
        aux_strength_state = Some(stats.strength);
    }
    if let Some(log_prec) = dim_selection_precision {
        if log_prec.len() != latent_dim {
            return Err(format!(
                "dim_selection_log_precision length {} must equal latent_dim {}",
                log_prec.len(),
                latent_dim
            ));
        }
        for a in 0..latent_dim {
            let log_alpha = log_prec[a];
            let alpha = log_alpha.exp();
            if !(alpha.is_finite() && alpha > 0.0) {
                return Err(format!(
                    "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                ));
            }
            let mut sq = 0.0_f64;
            for n in 0..n_obs {
                let v = t_mat[[n, a]];
                sq += v * v;
            }
            latent_prior_score += 0.5 * alpha * sq - 0.5 * (n_obs as f64) * log_alpha;
        }
    }
    Ok((latent_prior_score, aux_strength_state))
}

fn dense_fisher_gaussian_fit_to_pydict<'py>(
    py: Python<'py>,
    design: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fisher_w: ArrayView3<'_, f64>,
    init_lambda: Option<f64>,
    latent_prior_score: f64,
    aux_strength_state: Option<LatentAuxStrengthState>,
) -> PyResult<Py<PyDict>> {
    let n_obs = design.nrows();
    let k = design.ncols();
    let n_outputs = y.ncols();
    validate_dense_fisher_w(n_obs, n_outputs, fisher_w).map_err(py_value_error)?;
    if y.nrows() != n_obs {
        return Err(py_value_error(format!(
            "dense Fisher Gaussian row mismatch: X has {n_obs}, y has {}",
            y.nrows()
        )));
    }
    let row_weights = latent_row_weights(n_obs, weights).map_err(py_value_error)?;
    let lambda = init_lambda.unwrap_or(1.0);
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(py_value_error(format!(
            "init_lambda must be finite and positive for dense Fisher fit; got {lambda}"
        )));
    }
    let mut hessian = gam::pirls::dense_block_xtwx(design, fisher_w, Some(row_weights.view()))
        .map_err(|err| py_value_error(err.to_string()))?;
    add_block_diagonal_penalty(&mut hessian, penalty, lambda, n_outputs).map_err(py_value_error)?;
    let rhs = gam::pirls::dense_block_xtwy(design, fisher_w, y, Some(row_weights.view()))
        .map_err(|err| py_value_error(err.to_string()))?;
    let beta_vec = solve_dense_block_system(&hessian, &rhs, "dense Fisher Gaussian")
        .map_err(py_value_error)?;
    let mut coefficients = Array2::<f64>::zeros((k, n_outputs));
    for output in 0..n_outputs {
        for col in 0..k {
            coefficients[[col, output]] = beta_vec[output * k + col];
        }
    }
    let fitted = design.dot(&coefficients);
    let mut sigma2 = Array1::<f64>::zeros(n_outputs);
    let mut objective = latent_prior_score;
    for row in 0..n_obs {
        for a in 0..n_outputs {
            let ra = y[[row, a]] - fitted[[row, a]];
            sigma2[a] += row_weights[row] * ra * ra;
            for b in 0..n_outputs {
                objective += 0.5
                    * row_weights[row]
                    * ra
                    * fisher_w[[row, a, b]]
                    * (y[[row, b]] - fitted[[row, b]]);
            }
        }
    }
    for output in 0..n_outputs {
        sigma2[output] /= (n_obs.saturating_sub(k).max(1)) as f64;
        let beta_col = coefficients.column(output);
        let s_beta = penalty.dot(&beta_col);
        objective += 0.5 * lambda * beta_col.dot(&s_beta);
    }
    let out = PyDict::new(py);
    out.set_item("status", "ok")?;
    out.set_item("lambda", lambda)?;
    out.set_item("rho", lambda.ln())?;
    out.set_item("reml_score", objective)?;
    out.set_item("reml_grad_lambda", f64::NAN)?;
    out.set_item("reml_hess_lambda", f64::NAN)?;
    out.set_item("reml_grad_rho", f64::NAN)?;
    out.set_item("reml_hess_rho", f64::NAN)?;
    out.set_item("edf", (k * n_outputs) as f64)?;
    out.set_item("coefficients", coefficients.into_pyarray(py))?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("sigma2", sigma2.into_pyarray(py))?;
    out.set_item(
        "cache_penalty_eigenvalues",
        Array1::<f64>::zeros(0).into_pyarray(py),
    )?;
    out.set_item(
        "cache_eigenvectors",
        Array2::<f64>::zeros((0, 0)).into_pyarray(py),
    )?;
    out.set_item(
        "cache_coefficient_basis",
        Array2::<f64>::zeros((0, 0)).into_pyarray(py),
    )?;
    out.set_item("cache_xtwx_fingerprint", 0_u64)?;
    out.set_item("cache_penalty_fingerprint", 0_u64)?;
    out.set_item("cache_logdet_xtwx", f64::NAN)?;
    out.set_item("cache_logdet_penalty_positive", f64::NAN)?;
    out.set_item("cache_penalty_rank", 0_usize)?;
    out.set_item("cache_nullity", 0_usize)?;
    set_aux_strength_items(py, &out, aux_strength_state)?;
    Ok(out.unbind())
}

fn sigmoid_stable(eta: f64) -> f64 {
    if eta >= 0.0 {
        let e = (-eta).exp();
        1.0 / (1.0 + e)
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

fn softmax_with_baseline(eta_active: &[f64], out: &mut [f64]) {
    let mut max_eta = 0.0_f64;
    for &v in eta_active {
        max_eta = max_eta.max(v);
    }
    let mut denom = (-max_eta).exp();
    for (idx, &v) in eta_active.iter().enumerate() {
        let e = (v - max_eta).exp();
        out[idx] = e;
        denom += e;
    }
    for v in out.iter_mut().take(eta_active.len()) {
        *v /= denom;
    }
    out[eta_active.len()] = (-max_eta).exp() / denom;
}

fn dense_fisher_glm_fit_to_pydict<'py>(
    py: Python<'py>,
    design: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fisher_w: Option<ArrayView3<'_, f64>>,
    init_lambda: Option<f64>,
    family_name: &str,
    latent_prior_score: f64,
    aux_strength_state: Option<LatentAuxStrengthState>,
) -> PyResult<Py<PyDict>> {
    let n_obs = design.nrows();
    let k = design.ncols();
    let n_outputs = y.ncols();
    if y.nrows() != n_obs || n_outputs == 0 {
        return Err(py_value_error(format!(
            "dense Fisher GLM response shape mismatch: X has {n_obs} rows, y is {}x{}",
            y.nrows(),
            y.ncols()
        )));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(py_value_error(
            "dense Fisher GLM response must be finite".to_string(),
        ));
    }
    if let Some(fw) = fisher_w.as_ref() {
        validate_dense_fisher_w(n_obs, n_outputs, fw.view()).map_err(py_value_error)?;
    }
    let row_weights = latent_row_weights(n_obs, weights).map_err(py_value_error)?;
    let lambda = init_lambda.unwrap_or(1.0);
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(py_value_error(format!(
            "init_lambda must be finite and positive for dense Fisher GLM fit; got {lambda}"
        )));
    }
    let normalized = family_name.to_ascii_lowercase().replace('_', "-");
    let multinomial = matches!(
        normalized.as_str(),
        "multinomial" | "multinomial-logit" | "softmax" | "categorical-logit"
    );
    let binomial_multi = matches!(
        normalized.as_str(),
        "binomial" | "binomial-logit" | "logistic"
    );
    if multinomial && n_outputs < 2 {
        return Err(py_value_error(
            "multinomial-logit requires at least two response columns".to_string(),
        ));
    }
    if !(multinomial || binomial_multi) {
        return Err(py_value_error(format!(
            "dense multi-output GLM latent supports binomial-logit and multinomial-logit; got {family_name:?}"
        )));
    }

    let active_outputs = if multinomial {
        n_outputs - 1
    } else {
        n_outputs
    };
    let mut coefficients_active = Array2::<f64>::zeros((k, active_outputs));
    let mut fitted = Array2::<f64>::zeros((n_obs, n_outputs));
    let mut h_blocks = Array3::<f64>::zeros((n_obs, active_outputs, active_outputs));
    let mut gradient = Array1::<f64>::zeros(k * active_outputs);
    let max_iter = 50usize;
    let tol = 1.0e-7_f64;
    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 0..max_iter {
        iterations = iter + 1;
        gradient.fill(0.0);
        h_blocks.fill(0.0);
        if multinomial {
            let mut eta = vec![0.0_f64; active_outputs];
            let mut probs = vec![0.0_f64; n_outputs];
            for row in 0..n_obs {
                for a in 0..active_outputs {
                    let mut v = 0.0_f64;
                    for col in 0..k {
                        v += design[[row, col]] * coefficients_active[[col, a]];
                    }
                    eta[a] = v;
                }
                softmax_with_baseline(&eta, &mut probs);
                for a in 0..n_outputs {
                    fitted[[row, a]] = probs[a];
                }
                for a in 0..active_outputs {
                    let resid = probs[a] - y[[row, a]];
                    for col in 0..k {
                        gradient[a * k + col] += row_weights[row] * design[[row, col]] * resid;
                    }
                    for b in 0..active_outputs {
                        h_blocks[[row, a, b]] = if let Some(fw) = fisher_w.as_ref() {
                            fw[[row, a, b]]
                        } else if a == b {
                            probs[a] * (1.0 - probs[a])
                        } else {
                            -probs[a] * probs[b]
                        };
                    }
                }
            }
        } else {
            for row in 0..n_obs {
                for a in 0..active_outputs {
                    let mut eta = 0.0_f64;
                    for col in 0..k {
                        eta += design[[row, col]] * coefficients_active[[col, a]];
                    }
                    let mu = sigmoid_stable(eta);
                    fitted[[row, a]] = mu;
                    let resid = mu - y[[row, a]];
                    for col in 0..k {
                        gradient[a * k + col] += row_weights[row] * design[[row, col]] * resid;
                    }
                }
                for a in 0..active_outputs {
                    for b in 0..active_outputs {
                        h_blocks[[row, a, b]] = if let Some(fw) = fisher_w.as_ref() {
                            fw[[row, a, b]]
                        } else if a == b {
                            fitted[[row, a]] * (1.0 - fitted[[row, a]])
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        let mut hessian =
            gam::pirls::dense_block_xtwx(design, h_blocks.view(), Some(row_weights.view()))
                .map_err(|err| py_value_error(err.to_string()))?;
        add_block_diagonal_penalty(&mut hessian, penalty, lambda, active_outputs)
            .map_err(py_value_error)?;
        for a in 0..active_outputs {
            let beta_col = coefficients_active.column(a);
            let s_beta = penalty.dot(&beta_col);
            for col in 0..k {
                gradient[a * k + col] += lambda * s_beta[col];
            }
        }
        let rhs = gradient.mapv(|v| -v);
        let delta =
            solve_dense_block_system(&hessian, &rhs, "dense Fisher GLM").map_err(py_value_error)?;
        let mut step_norm = 0.0_f64;
        for a in 0..active_outputs {
            for col in 0..k {
                let d = delta[a * k + col];
                coefficients_active[[col, a]] += d;
                step_norm += d * d;
            }
        }
        if step_norm.sqrt()
            <= tol
                * (1.0
                    + coefficients_active
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt())
        {
            converged = true;
            break;
        }
    }

    let mut objective = latent_prior_score;
    if multinomial {
        for row in 0..n_obs {
            for a in 0..n_outputs {
                objective -= row_weights[row] * y[[row, a]] * fitted[[row, a]].max(1.0e-12).ln();
            }
        }
    } else {
        for row in 0..n_obs {
            for a in 0..n_outputs {
                let mu = fitted[[row, a]].clamp(1.0e-12, 1.0 - 1.0e-12);
                objective -= row_weights[row]
                    * (y[[row, a]] * mu.ln() + (1.0 - y[[row, a]]) * (1.0 - mu).ln());
            }
        }
    }
    for a in 0..active_outputs {
        let beta_col = coefficients_active.column(a);
        let s_beta = penalty.dot(&beta_col);
        objective += 0.5 * lambda * beta_col.dot(&s_beta);
    }
    let mut coefficients = Array2::<f64>::zeros((k, n_outputs));
    for a in 0..active_outputs {
        for col in 0..k {
            coefficients[[col, a]] = coefficients_active[[col, a]];
        }
    }
    let out = PyDict::new(py);
    out.set_item("status", if converged { "ok" } else { "not_converged" })?;
    out.set_item("lambda", lambda)?;
    out.set_item("rho", lambda.ln())?;
    out.set_item("reml_score", objective)?;
    out.set_item("reml_grad_lambda", f64::NAN)?;
    out.set_item("reml_hess_lambda", f64::NAN)?;
    out.set_item("reml_grad_rho", f64::NAN)?;
    out.set_item("reml_hess_rho", f64::NAN)?;
    out.set_item("edf", (k * active_outputs) as f64)?;
    out.set_item("coefficients", coefficients.into_pyarray(py))?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("sigma2", Array1::<f64>::ones(n_outputs).into_pyarray(py))?;
    out.set_item("iterations", iterations)?;
    out.set_item(
        "cache_penalty_eigenvalues",
        Array1::<f64>::zeros(0).into_pyarray(py),
    )?;
    out.set_item(
        "cache_eigenvectors",
        Array2::<f64>::zeros((0, 0)).into_pyarray(py),
    )?;
    out.set_item(
        "cache_coefficient_basis",
        Array2::<f64>::zeros((0, 0)).into_pyarray(py),
    )?;
    out.set_item("cache_xtwx_fingerprint", 0_u64)?;
    out.set_item("cache_penalty_fingerprint", 0_u64)?;
    out.set_item("cache_logdet_xtwx", f64::NAN)?;
    out.set_item("cache_logdet_penalty_positive", f64::NAN)?;
    out.set_item("cache_penalty_rank", 0_usize)?;
    out.set_item("cache_nullity", 0_usize)?;
    set_aux_strength_items(py, &out, aux_strength_state)?;
    Ok(out.unbind())
}

fn latent_augmented_hessian_factor(
    design: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    lambda: f64,
) -> Result<gam::faer_ndarray::FaerSymmetricFactor, String> {
    if penalty.dim() != (design.ncols(), design.ncols()) {
        return Err(format!(
            "penalty shape mismatch for latent Hessian: expected {}x{}, got {}x{}",
            design.ncols(),
            design.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    let mut hessian = fast_xt_diag_x(&design, &weights);
    for row in 0..hessian.nrows() {
        for col in 0..hessian.ncols() {
            let s_sym = 0.5 * (penalty[[row, col]] + penalty[[col, row]]);
            hessian[[row, col]] += lambda * s_sym;
        }
    }
    factorize_symmetricwith_fallback(
        gam::faer_ndarray::FaerArrayView::new(&hessian).as_ref(),
        Side::Lower,
    )
    .map_err(|err| format!("latent REML Hessian factorization failed: {err}"))
}

/// Add the analytic outer REML score contribution to `grad_t`.
///
/// Implements `/tmp/codex_outer_analytic.md`:
///
/// ```text
/// ∇_{t_i} V = w_i J_i^T [K_H x_i - (r_i / σ_eff²) β]
/// ```
///
/// This is the Occam-factor contribution called out in the
/// `composition_engine.md` §7 audit revisions. The previous
/// `gaussian_reml_fit_latent_backward` path contracted only the data-fit
/// design gradient for the latent row; without the `K_H x_i` term, the
/// returned `grad_t` omitted the REML log-determinant correction. `K_H` is
/// never materialized here: the row vector `u_i = K_H x_i` is computed by one
/// solve against the shared factor of `H = X^T W X + λS` per row and then
/// reused across all latent coordinates for that row.
///
/// Future `S(ρ, t)` penalties belong after the row-local contraction below:
/// add the standard `β^T S_{i,a} β`, `tr(K_H S_{i,a})`, and
/// `tr(S_+ S_{i,a})` terms from the derivation when coefficient penalties
/// become input-location dependent.
fn add_latent_outer_reml_score_gradient(
    grad_t: &mut Array1<f64>,
    scale: f64,
    design: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    t_mat: ArrayView2<'_, f64>,
    jet: &Array3<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fit: &gam::gaussian_reml::GaussianRemlMultiResult,
    sigma_eff_mode: SigmaEffMode,
) -> Result<(), String> {
    if scale == 0.0 {
        return Ok(());
    }
    let n_obs = design.nrows();
    let p = design.ncols();
    let latent_dim = t_mat.ncols();
    let n_outputs = fit.coefficients.ncols();
    if grad_t.len() != n_obs * latent_dim {
        return Err(format!(
            "latent grad_t shape mismatch: expected {}, got {}",
            n_obs * latent_dim,
            grad_t.len()
        ));
    }
    if fit.coefficients.nrows() != p || fit.fitted.nrows() != n_obs {
        return Err("latent REML fit shape mismatch".to_string());
    }
    if y.dim() != (n_obs, n_outputs) {
        return Err(format!(
            "latent REML response shape mismatch: expected {}x{}, got {}x{}",
            n_obs,
            n_outputs,
            y.nrows(),
            y.ncols()
        ));
    }
    let weight = gaussian_reml_weight_vector_local(n_obs, weights)?;
    let factor = latent_augmented_hessian_factor(design, penalty, weight.view(), fit.lambda)?;
    if jet.shape() != [n_obs, p, latent_dim] {
        return Err(format!(
            "latent REML jet shape mismatch: expected {}x{}x{}, got {}x{}x{}",
            n_obs,
            p,
            latent_dim,
            jet.shape()[0],
            jet.shape()[1],
            jet.shape()[2],
        ));
    }
    let mut rhs = Array2::<f64>::zeros((p, 1));
    let mut direction = Array1::<f64>::zeros(p);
    for n in 0..n_obs {
        for col in 0..p {
            rhs[[col, 0]] = design[[n, col]];
        }
        {
            let mut rhs_view = array2_to_matmut(&mut rhs);
            factor.solve_in_place(rhs_view.as_mut());
        }
        for col in 0..p {
            direction[col] = (n_outputs as f64) * rhs[[col, 0]];
        }
        for output in 0..n_outputs {
            let sigma_eff2 = match sigma_eff_mode {
                SigmaEffMode::Profiled => fit.sigma2[output],
                // Fixed-dispersion latent REML is wired as an explicit mode so
                // the Python/Rust boundary is stable. The current closed-form
                // forward does not accept an external σ² yet, so the fixed path
                // uses the fit's σ² slot until that forward parameter lands.
                SigmaEffMode::Fixed => fit.sigma2[output],
            };
            if !(sigma_eff2.is_finite() && sigma_eff2 > 0.0) {
                return Err(format!(
                    "sigma_eff2 must be finite and positive; got {sigma_eff2}"
                ));
            }
            let residual = y[[n, output]] - fit.fitted[[n, output]];
            let data_scale = residual / sigma_eff2;
            for col in 0..p {
                direction[col] -= data_scale * fit.coefficients[[col, output]];
            }
        }
        let row_scale = scale * weight[n];
        for col in 0..p {
            let d_col = direction[col];
            if d_col == 0.0 {
                continue;
            }
            let col_scale = row_scale * d_col;
            for a in 0..latent_dim {
                grad_t[n * latent_dim + a] += col_scale * jet[[n, col, a]];
            }
        }
    }
    Ok(())
}

fn analytic_penalty_value_for_targets(
    registry: &AnalyticPenaltyRegistry,
    target_t: ArrayView1<'_, f64>,
    target_beta: Option<ArrayView1<'_, f64>>,
) -> Result<f64, String> {
    let rho = Array1::<f64>::zeros(registry.total_rho_count());
    let mut value = 0.0_f64;
    for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(registry.rho_layout())
    {
        let rho_local = rho.slice(s![rho_slice]);
        let target = match (tier, penalty) {
            (PenaltyTier::Psi, _)
            | (PenaltyTier::Beta, AnalyticPenaltyKind::MechanismSparsity(_)) => target_t.view(),
            (PenaltyTier::Beta, _) => {
                let Some(target_beta) = target_beta.as_ref() else {
                    continue;
                };
                target_beta.view()
            }
            (PenaltyTier::Rho, _) => continue,
        };
        value += penalty.value(target, rho_local);
    }
    Ok(value)
}

fn gaussian_reml_fit_latent_impl(
    t_flat: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
    basis_kind: &str,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<ArrayView2<'_, f64>>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
    dim_selection_precision: Option<ArrayView1<'_, f64>>,
    analytic_penalties: Option<&AnalyticPenaltyRegistry>,
) -> Result<
    (
        gam::gaussian_reml::GaussianRemlMultiResult,
        Array2<f64>,
        Option<LatentAuxStrengthState>,
    ),
    String,
> {
    let (design, t_mat, _jet) = build_latent_forward_design(
        basis_kind,
        t_flat,
        n_obs,
        latent_dim,
        centers,
        m,
        tensor_knots_concat,
        tensor_knot_offsets,
        tensor_degrees,
    )?;
    // Build the (optionally) augmented Y/X stack carrying the identifiability
    // penalty. The penalty `½ μ ‖t − t_ref‖²` is *not* on the design Φ; it
    // acts on t directly. Because t enters Φ nonlinearly, we cannot fold it
    // into the inner Gaussian-closed-form solve without changing the solver.
    // We therefore evaluate the *penalty contribution* here and return it
    // for the caller to expose; the inner ridge stays unchanged.
    //
    // The forward path's responsibility is to produce a self-consistent fit
    // at the current t; the outer loop owns the gauge enforcement (it adds
    // ∂R_id/∂t to grad_t and walks t under that combined gradient).
    let mut fit = gaussian_reml_multi_closed_form_with_cache(
        design.view(),
        y,
        penalty,
        weights,
        init_lambda,
        None,
    )
    .map_err(|err| err.to_string())?;
    // Fixes audit-revised claim that ARD / aux-prior REML selection requires
    // normalized priors, not raw quadratic corrections alone.
    let mut latent_prior_score = 0.0_f64;
    let mut aux_strength_state = None;
    if let Some(u_view) = aux_u {
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, aux_family, aux_strength)?;
        latent_prior_score += stats.score;
        aux_strength_state = Some(stats.strength);
    }
    if let Some(log_prec) = dim_selection_precision {
        if log_prec.len() != latent_dim {
            return Err(format!(
                "dim_selection_log_precision length {} must equal latent_dim {}",
                log_prec.len(),
                latent_dim
            ));
        }
        for a in 0..latent_dim {
            let log_alpha = log_prec[a];
            let alpha = log_alpha.exp();
            if !(alpha.is_finite() && alpha > 0.0) {
                return Err(format!(
                    "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                ));
            }
            let mut sq = 0.0_f64;
            for n in 0..n_obs {
                let v = t_mat[[n, a]];
                sq += v * v;
            }
            latent_prior_score += 0.5 * alpha * sq - 0.5 * (n_obs as f64) * log_alpha;
        }
    }
    if let Some(registry) = analytic_penalties {
        latent_prior_score += analytic_penalty_value_for_targets(registry, t_flat, None)?;
    }
    fit.reml_score += latent_prior_score;
    Ok((fit, design, aux_strength_state))
}

/// Forward fit: build the latent design at the current latent `t`,
/// solve the Gaussian REML inner problem, and return the standard
/// REML fit dictionary plus the materialized design (for warm-starts).
///
/// `t` is a flat row-major `(n_obs * latent_dim)` array; `n_obs` and
/// `latent_dim` must be passed explicitly because the flat vector cannot
/// carry shape.
#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    analytic_penalties = None,
    basis_kind = "duchon".to_string(),
    tensor_knots_concat = None,
    tensor_knot_offsets = None,
    tensor_degrees = None,
))]
fn gaussian_reml_fit_latent<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    analytic_penalties: Option<String>,
    basis_kind: String,
    tensor_knots_concat: Option<PyReadonlyArray1<'py, f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
) -> PyResult<Py<PyDict>> {
    let analytic_penalties: Option<serde_json::Value> = match analytic_penalties {
        Some(s) => Some(serde_json::from_str(&s).map_err(|e| py_value_error(e.to_string()))?),
        None => None,
    };
    let latent_payload = serde_json::json!({"t": {"name": "t", "n": n_obs, "d": latent_dim}});
    let registry = build_analytic_penalty_registry_from_json(
        Some(&latent_payload),
        analytic_penalties.as_ref(),
    )
    .map_err(py_value_error)?;
    let family = match aux_family.to_ascii_lowercase().as_str() {
        "ridge" => AuxPriorFamily::Ridge,
        "linear" => AuxPriorFamily::Linear,
        other => {
            return Err(py_value_error(format!(
                "aux_family must be 'ridge' or 'linear'; got {other:?}"
            )));
        }
    };
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let centers_values = centers.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let fisher_values = fisher_w.as_ref().map(|w| w.as_array().to_owned());
    let aux_u_values = aux_u.as_ref().map(|a| a.as_array().to_owned());
    let dim_selection_values = dim_selection_log_precision
        .as_ref()
        .map(|a| a.as_array().to_owned());
    let tensor_knots_values = tensor_knots_concat
        .as_ref()
        .map(|a| a.as_array().to_owned());
    if let Some(fw) = fisher_values.as_ref() {
        let n_outputs = y_values.ncols();
        if fw.shape() == [n_obs, n_outputs, n_outputs] && n_outputs > 1 {
            let (design, t_mat, _jet) = build_latent_forward_design(
                &basis_kind,
                t_values.view(),
                n_obs,
                latent_dim,
                centers_values.view(),
                m,
                tensor_knots_values.as_ref().map(|a| a.view()),
                tensor_knot_offsets.as_deref(),
                tensor_degrees.as_deref(),
            )
            .map_err(py_value_error)?;
            let (prior_score, aux_strength_state) = latent_prior_score_and_aux_state_for_t(
                t_mat.view(),
                aux_u_values.as_ref().map(|a| a.view()),
                family,
                aux_strength,
                dim_selection_values.as_ref().map(|a| a.view()),
            )
            .map_err(py_value_error)?;
            let analytic_score =
                analytic_penalty_value_for_targets(&registry, t_values.view(), None)
                    .map_err(py_value_error)?;
            return dense_fisher_gaussian_fit_to_pydict(
                py,
                design.view(),
                y_values.view(),
                penalty_values.view(),
                weight_values.as_ref().map(|w| w.view()),
                fw.view(),
                init_lambda,
                prior_score + analytic_score,
                aux_strength_state,
            );
        }
    }
    let analytic_penalties_for_thread = analytic_penalties.clone();
    let latent_payload_for_thread = latent_payload.clone();
    let (fit, _design, aux_strength_state) =
        detach_py_result(py, "gaussian_reml_fit_latent", move || {
            let registry = build_analytic_penalty_registry_from_json(
                Some(&latent_payload_for_thread),
                analytic_penalties_for_thread.as_ref(),
            )?;
            let effective_weights = latent_scalar_weights_with_fisher(
                n_obs,
                weight_values.as_ref().map(|w| w.view()),
                fisher_values.as_ref().map(|w| w.view()),
            )?;
            gaussian_reml_fit_latent_impl(
                t_values.view(),
                y_values.view(),
                n_obs,
                latent_dim,
                centers_values.view(),
                m,
                &basis_kind,
                tensor_knots_values.as_ref().map(|a| a.view()),
                tensor_knot_offsets.as_deref(),
                tensor_degrees.as_deref(),
                penalty_values.view(),
                effective_weights.as_ref().map(|w| w.view()),
                init_lambda,
                aux_u_values.as_ref().map(|a| a.view()),
                family,
                aux_strength,
                dim_selection_values.as_ref().map(|a| a.view()),
                Some(&registry),
            )
        })?;
    let out = PyDict::new(py);
    set_ok_gaussian_reml_items(py, &out, fit)?;
    set_aux_strength_items(py, &out, aux_strength_state)?;
    Ok(out.unbind())
}

fn sae_atom_basis_kind_from_str(value: &str) -> SaeAtomBasisKind {
    match value.to_ascii_lowercase().replace('-', "_").as_str() {
        "duchon" => SaeAtomBasisKind::Duchon,
        "periodic" | "periodic_spline" | "circle" => SaeAtomBasisKind::Periodic,
        "sphere" => SaeAtomBasisKind::Sphere,
        "euclidean" | "euclidean_patch" => SaeAtomBasisKind::EuclideanPatch,
        other => SaeAtomBasisKind::Precomputed(other.to_string()),
    }
}

fn gumbel_temperature_schedule_from_pydict(
    schedule: Option<&Bound<'_, PyDict>>,
) -> Result<Option<GumbelTemperatureSchedule>, String> {
    fn get<'py>(state: &'py Bound<'py, PyDict>, key: &str) -> Result<Bound<'py, PyAny>, String> {
        state
            .get_item(key)
            .map_err(|err| err.to_string())?
            .ok_or_else(|| format!("gumbel_schedule is missing key {key:?}"))
    }

    let Some(schedule) = schedule else {
        return Ok(None);
    };
    let decay_name = get(schedule, "decay")?
        .extract::<String>()
        .map_err(|err| err.to_string())?
        .to_ascii_lowercase()
        .replace('-', "_");
    let tau_start = get(schedule, "tau_start")?
        .extract::<f64>()
        .map_err(|err| err.to_string())?;
    let tau_min = match schedule
        .get_item("tau_min")
        .map_err(|err| err.to_string())?
        .or(schedule
            .get_item("tau_end")
            .map_err(|err| err.to_string())?)
    {
        Some(value) => value.extract::<f64>().map_err(|err| err.to_string())?,
        None => return Err("gumbel_schedule is missing key \"tau_min\"".to_string()),
    };
    let decay = match decay_name.as_str() {
        "geometric" | "exponential" => {
            let rate = match schedule.get_item("rate").map_err(|err| err.to_string())? {
                Some(value) => value.extract::<f64>().map_err(|err| err.to_string())?,
                None => 0.9,
            };
            ScheduleKind::Geometric { rate }
        }
        "linear" => {
            let steps = get(schedule, "steps")?
                .extract::<usize>()
                .map_err(|err| err.to_string())?;
            ScheduleKind::Linear { steps }
        }
        "reciprocal_iter" => ScheduleKind::ReciprocalIter,
        other => {
            return Err(format!(
                "gumbel_schedule decay must be 'geometric', 'exponential', 'linear', or 'reciprocal_iter'; got {other:?}"
            ));
        }
    };
    let mut schedule_out = GumbelTemperatureSchedule::new(tau_start, tau_min, decay)?;
    if let Some(iter_count) = schedule
        .get_item("iter_count")
        .map_err(|err| err.to_string())?
    {
        schedule_out.iter_count = iter_count
            .extract::<usize>()
            .map_err(|err| err.to_string())?;
        schedule_out.validate()?;
    }
    Ok(Some(schedule_out))
}

/// Fit one SAE-manifold refresh step; auto lambda selection is expressed by
/// passing `lambda_grid`, which evaluates candidate smoothness lambdas in Rust.
#[pyfunction(signature = (
    z,
    atom_basis,
    atom_dim,
    basis_values,
    basis_jacobian,
    basis_sizes,
    decoder_coefficients,
    smooth_penalties,
    initial_logits,
    initial_coords,
    alpha,
    tau,
    learnable_alpha,
    assignment_kind,
    sparsity_strength = 1.0,
    smoothness = 1.0,
    lambda_grid = None,
    max_iter = 12,
    learning_rate = 1.0,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    gumbel_schedule = None,
    analytic_penalties = None,
))]
fn sae_manifold_fit<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    basis_values: PyReadonlyArray3<'py, f64>,
    basis_jacobian: PyReadonlyArray4<'py, f64>,
    basis_sizes: Vec<usize>,
    decoder_coefficients: PyReadonlyArray3<'py, f64>,
    smooth_penalties: PyReadonlyArray3<'py, f64>,
    initial_logits: PyReadonlyArray2<'py, f64>,
    initial_coords: PyReadonlyArray3<'py, f64>,
    alpha: f64,
    tau: f64,
    learnable_alpha: bool,
    assignment_kind: String,
    sparsity_strength: f64,
    smoothness: f64,
    lambda_grid: Option<Vec<f64>>,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    gumbel_schedule: Option<&Bound<'py, PyDict>>,
    analytic_penalties: Option<String>,
) -> PyResult<Py<PyDict>> {
    let analytic_penalties: Option<serde_json::Value> = match analytic_penalties {
        Some(s) => Some(serde_json::from_str(&s).map_err(|e| py_value_error(e.to_string()))?),
        None => None,
    };
    let z_view = z.as_array();
    let (n_obs, p_out) = z_view.dim();
    if n_obs == 0 || p_out == 0 {
        return Err(py_value_error(
            "sae_manifold_fit requires a non-empty (N, p) response".to_string(),
        ));
    }
    let k_atoms = atom_dim.len();
    if k_atoms == 0 {
        return Err(py_value_error(
            "sae_manifold_fit requires at least one atom".to_string(),
        ));
    }
    if atom_basis.len() != k_atoms || basis_sizes.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_manifold_fit metadata lengths must equal K={k_atoms}; got atom_basis={}, basis_sizes={}",
            atom_basis.len(),
            basis_sizes.len()
        )));
    }
    let latent_payload = serde_json::json!({"t": {"name": "t", "n": n_obs, "d": atom_dim.iter().copied().max().unwrap_or(1)}});
    let registry = build_analytic_penalty_registry_from_json(
        Some(&latent_payload),
        analytic_penalties.as_ref(),
    )
    .map_err(py_value_error)?;
    if max_iter != 1 {
        return Err(py_value_error(
            "sae_manifold_fit accepts exactly one Newton step per basis snapshot; the Python driver refreshes Phi and dPhi/dt between calls".to_string(),
        ));
    }
    if initial_logits.as_array().dim() != (n_obs, k_atoms) {
        return Err(py_value_error(format!(
            "initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
            initial_logits.as_array().dim()
        )));
    }
    for (name, value) in [
        ("alpha", alpha),
        ("tau", tau),
        ("sparsity_strength", sparsity_strength),
        ("smoothness", smoothness),
        ("learning_rate", learning_rate),
        ("ridge_ext_coord", ridge_ext_coord),
        ("ridge_beta", ridge_beta),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(py_value_error(format!(
                "{name} must be finite and positive; got {value}"
            )));
        }
    }
    let lambda_candidates = match lambda_grid {
        Some(grid) => {
            if grid.is_empty() {
                return Err(py_value_error(
                    "lambda_grid must contain at least one candidate".to_string(),
                ));
            }
            for (idx, value) in grid.iter().copied().enumerate() {
                if !value.is_finite() || value <= 0.0 {
                    return Err(py_value_error(format!(
                        "lambda_grid[{idx}] must be finite and positive; got {value}"
                    )));
                }
            }
            grid
        }
        None => vec![smoothness],
    };

    let basis_values_shape = basis_values.as_array().shape().to_vec();
    if basis_values_shape[0] != k_atoms || basis_values_shape[1] != n_obs {
        return Err(py_value_error(format!(
            "basis_values must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_values_shape
        )));
    }
    let basis_jacobian_shape = basis_jacobian.as_array().shape().to_vec();
    if basis_jacobian_shape[0] != k_atoms || basis_jacobian_shape[1] != n_obs {
        return Err(py_value_error(format!(
            "basis_jacobian must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_jacobian_shape
        )));
    }
    let decoder_shape = decoder_coefficients.as_array().shape().to_vec();
    if decoder_shape[0] != k_atoms || decoder_shape[2] != p_out {
        return Err(py_value_error(format!(
            "decoder_coefficients must have shape (K, M_max, p)=({k_atoms}, M_max, {p_out}); got {:?}",
            decoder_shape
        )));
    }
    let smooth_shape = smooth_penalties.as_array().shape().to_vec();
    if smooth_shape[0] != k_atoms || smooth_shape[1] != smooth_shape[2] {
        return Err(py_value_error(format!(
            "smooth_penalties must have shape (K, M_max, M_max); got {:?}",
            smooth_shape
        )));
    }
    let coords_view = initial_coords.as_array();
    let coords_shape = coords_view.shape().to_vec();
    if coords_shape[0] != k_atoms || coords_shape[1] != n_obs {
        return Err(py_value_error(format!(
            "initial_coords must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            coords_shape
        )));
    }
    let max_dim = coords_view.shape().get(2).copied().ok_or_else(|| {
        py_value_error("initial_coords must be a rank-3 (K, N, D_max) array".to_string())
    })?;
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let d = atom_dim[atom_idx];
        let m = basis_sizes[atom_idx];
        if m > basis_values_shape[2]
            || m > basis_jacobian_shape[2]
            || m > decoder_shape[1]
            || m > smooth_shape[1]
        {
            return Err(py_value_error(format!(
                "basis_sizes[{atom_idx}]={m} exceeds one of the padded M_max dimensions"
            )));
        }
        if d > max_dim {
            return Err(py_value_error(format!(
                "atom_dim[{atom_idx}]={d} exceeds initial_coords D_max={max_dim}"
            )));
        }
        if d > basis_jacobian_shape[3] {
            return Err(py_value_error(format!(
                "atom_dim[{atom_idx}]={d} exceeds basis_jacobian D_max={}",
                basis_jacobian_shape[3]
            )));
        }
        coord_blocks.push(coords_view.slice(s![atom_idx, 0..n_obs, 0..d]).to_owned());
    }

    let basis_kinds: Vec<SaeAtomBasisKind> = atom_basis
        .iter()
        .map(|kind| sae_atom_basis_kind_from_str(kind))
        .collect();
    let mode = match assignment_kind.as_str() {
        "softmax" => AssignmentMode::softmax(tau),
        "ibp_map" => AssignmentMode::ibp_map(tau, alpha, learnable_alpha),
        "jumprelu" => AssignmentMode::jumprelu(tau, tau),
        _ => {
            return Err(py_value_error(format!(
                "assignment_kind must be one of 'softmax', 'ibp_map', or 'jumprelu'; got {assignment_kind}"
            )));
        }
    };
    let mut base_term = term_from_padded_blocks_with_mode(
        n_obs,
        p_out,
        &basis_kinds,
        basis_values.as_array(),
        basis_jacobian.as_array(),
        &basis_sizes,
        &atom_dim,
        decoder_coefficients.as_array(),
        smooth_penalties.as_array(),
        initial_logits.as_array(),
        &coord_blocks,
        mode,
    )
    .map_err(py_value_error)?;
    if let Some(schedule) =
        gumbel_temperature_schedule_from_pydict(gumbel_schedule).map_err(py_value_error)?
    {
        base_term
            .set_temperature_schedule(schedule)
            .map_err(py_value_error)?;
    }

    let log_ard: Vec<Array1<f64>> = atom_dim
        .iter()
        .map(|&d| Array1::<f64>::zeros(d))
        .collect();
    let mut best_term = None;
    let mut best_rho = None;
    let mut best_loss = None;
    let mut best_score = f64::NEG_INFINITY;
    for lambda_smooth in lambda_candidates {
        let mut candidate_term = base_term.clone();
        let mut candidate_rho =
            SaeManifoldRho::new(sparsity_strength.ln(), lambda_smooth.ln(), log_ard.clone());
        let candidate_loss = candidate_term
            .run_single_external_basis_refresh_step_arrow_schur(
                z_view,
                &mut candidate_rho,
                Some(&registry),
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
            .map_err(py_value_error)?;
        let candidate_score = candidate_loss.evidence_proxy();
        if candidate_score > best_score {
            best_score = candidate_score;
            best_term = Some(candidate_term);
            best_rho = Some(candidate_rho);
            best_loss = Some(candidate_loss);
        }
    }
    let term = best_term.ok_or_else(|| {
        py_value_error("lambda_grid must contain at least one candidate".to_string())
    })?;
    let rho = best_rho.ok_or_else(|| {
        py_value_error("lambda_grid must contain at least one candidate".to_string())
    })?;
    let loss = best_loss.ok_or_else(|| {
        py_value_error("lambda_grid must contain at least one candidate".to_string())
    })?;

    let assignments = term.assignment.assignments();
    let fitted = term.fitted();
    let log_ard_py = PyList::empty(py);
    for atom_log_ard in &rho.log_ard {
        log_ard_py.append(atom_log_ard.clone().into_pyarray(py))?;
    }
    let atoms_py = PyList::empty(py);
    for atom_idx in 0..k_atoms {
        let atom = &term.atoms[atom_idx];
        let atom_dict = PyDict::new(py);
        atom_dict.set_item(
            "decoder_B",
            atom.decoder_coefficients.clone().into_pyarray(py),
        )?;
        atom_dict.set_item("basis_kind", atom_basis[atom_idx].clone())?;
        atom_dict.set_item("basis_centers", py.None())?;
        atom_dict.set_item(
            "on_atom_coords_t",
            term.assignment.coords[atom_idx]
                .as_matrix()
                .into_pyarray(py),
        )?;
        atom_dict.set_item(
            "assignments_z",
            assignments.column(atom_idx).to_owned().into_pyarray(py),
        )?;
        atom_dict.set_item("active_dim", atom_dim[atom_idx])?;
        atoms_py.append(atom_dict)?;
    }

    let active_mask: Vec<bool> = (0..k_atoms)
        .map(|atom_idx| assignments.column(atom_idx).sum() > 1.0e-8)
        .collect();
    let out = PyDict::new(py);
    out.set_item("atoms", atoms_py)?;
    out.set_item("assignments_z", assignments.into_pyarray(py))?;
    out.set_item("logits", term.assignment.logits.clone().into_pyarray(py))?;
    out.set_item("atom_active_mask", active_mask)?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("reml_score", loss.evidence_proxy())?;
    out.set_item(
        "log_alpha",
        alpha.ln()
            + if learnable_alpha {
                rho.log_lambda_sparse
            } else {
                0.0
            },
    )?;
    out.set_item("log_lambda_smooth", rho.log_lambda_smooth)?;
    out.set_item("log_ard", log_ard_py)?;
    out.set_item("assignment_prior", assignment_kind)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (
    z,
    atom_basis,
    atom_dim,
    basis_values,
    basis_jacobian,
    basis_sizes,
    decoder_coefficients,
    smooth_penalties,
    initial_logits,
    initial_coords,
    alpha,
    tau,
    learnable_alpha,
    sparsity_strength = 1.0,
    smoothness = 1.0,
    lambda_grid = None,
    max_iter = 12,
    learning_rate = 1.0,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    gumbel_schedule = None,
    analytic_penalties = None,
))]
fn sae_manifold_fit_ibp<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    basis_values: PyReadonlyArray3<'py, f64>,
    basis_jacobian: PyReadonlyArray4<'py, f64>,
    basis_sizes: Vec<usize>,
    decoder_coefficients: PyReadonlyArray3<'py, f64>,
    smooth_penalties: PyReadonlyArray3<'py, f64>,
    initial_logits: PyReadonlyArray2<'py, f64>,
    initial_coords: PyReadonlyArray3<'py, f64>,
    alpha: f64,
    tau: f64,
    learnable_alpha: bool,
    sparsity_strength: f64,
    smoothness: f64,
    lambda_grid: Option<Vec<f64>>,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    gumbel_schedule: Option<&Bound<'py, PyDict>>,
    analytic_penalties: Option<String>,
) -> PyResult<Py<PyDict>> {
    sae_manifold_fit(
        py,
        z,
        atom_basis,
        atom_dim,
        basis_values,
        basis_jacobian,
        basis_sizes,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        alpha,
        tau,
        learnable_alpha,
        "ibp_map".to_string(),
        sparsity_strength,
        smoothness,
        lambda_grid,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        gumbel_schedule,
        analytic_penalties,
    )
}

#[pyfunction(signature = (x, w_gate, w_amp))]
fn gated_sae_decode<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w_gate: PyReadonlyArray2<'py, f64>,
    w_amp: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let decoder = GatedSAEDecoder::new(w_gate.as_array().to_owned(), w_amp.as_array().to_owned())
        .map_err(py_value_error)?;
    let out = decoder.decode_batch(x.as_array()).map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Backward pass: compute `grad_t` and the standard REML adjoint
/// gradients at the current latent `t`.
///
/// The construction mirrors `gaussian_reml_fit_positions_backward`:
/// the inner adjoint produces `grad_x` (= ∂L/∂Φ); we then contract
/// against the N-D radial derivative jet to obtain
/// `grad_t ∈ ℝ^{n_obs × latent_dim}`.
///
/// For `grad_reml_score`, the latent contraction uses the explicit outer
/// REML formula from `/tmp/codex_outer_analytic.md` so the REML Occam
/// correction `J_i^T K_H x_i` is included with one shared solve per row.
///
/// Identifiability-mode contributions to `grad_t`:
///   * `AuxPrior`: the projected pullback of `μ · (t − ĥ(u))`;
///   * `DimSelection`: `+ Λ · t` with diagonal precision per axis.
///
/// These additive terms are computed here from the supplied auxiliary /
/// precision arrays and folded into the returned `grad_t`. The outer
/// REML loop sees a *unique* minimum because the inner Hessian on t is
/// now bounded below by `μI` (auxiliary). Fixes the audit-revised claim:
/// dim-selection/ARD alone is not a rotation-gauge fix and must be paired
/// with AuxPrior or Isometry for identifiability.
#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    grad_lambda = 0.0,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = 0.0,
    grad_edf = 0.0,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    basis_kind = "duchon".to_string(),
    sigma_eff_mode = "profiled".to_string(),
    tensor_knots_concat = None,
    tensor_knot_offsets = None,
    tensor_degrees = None,
))]
fn gaussian_reml_fit_latent_backward<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    grad_lambda: f64,
    grad_coefficients: Option<PyReadonlyArray2<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: f64,
    grad_edf: f64,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    basis_kind: String,
    sigma_eff_mode: String,
    tensor_knots_concat: Option<PyReadonlyArray1<'py, f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
) -> PyResult<Py<PyDict>> {
    let family = match aux_family.to_ascii_lowercase().as_str() {
        "ridge" => AuxPriorFamily::Ridge,
        "linear" => AuxPriorFamily::Linear,
        other => {
            return Err(py_value_error(format!(
                "aux_family must be 'ridge' or 'linear'; got {other:?}"
            )));
        }
    };
    let sigma_eff_mode = SigmaEffMode::parse(&sigma_eff_mode).map_err(py_value_error)?;
    let basis_kind_normalized = latent_basis_kind(&basis_kind).map_err(py_value_error)?;
    let centers_view = centers.as_array();
    let t_view = t.as_array();
    let y_view = y.as_array();
    let penalty_view = penalty.as_array();
    let effective_weights = latent_scalar_weights_with_fisher(
        n_obs,
        weights.as_ref().map(|w| w.as_array()),
        fisher_w.as_ref().map(|w| w.as_array()),
    )
    .map_err(py_value_error)?;
    let weights_view = effective_weights.as_ref().map(|w| w.view());

    // Forward design (Φ), t-matrix, and input-location jet share one dispatcher.
    let (design, t_mat, jet) = build_latent_forward_design(
        basis_kind_normalized,
        t_view,
        n_obs,
        latent_dim,
        centers_view,
        m,
        tensor_knots_concat.as_ref().map(|a| a.as_array()),
        tensor_knot_offsets.as_deref(),
        tensor_degrees.as_deref(),
    )
    .map_err(py_value_error)?;
    let fit = gaussian_reml_multi_closed_form_with_cache(
        design.view(),
        y_view,
        penalty_view,
        weights_view,
        init_lambda,
        None,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    // Inner adjoint for the returned standard gradients. This still follows
    // the generic REML VJP path for y/S/w, but grad_t below replaces the
    // score-design component with the row-shared analytic latent formula.
    let backward = gaussian_reml_multi_closed_form_backward_from_fit(
        design.view(),
        y_view,
        penalty_view,
        weights_view,
        &fit,
        grad_lambda,
        grad_coefficients.as_ref().map(|g| g.as_array()),
        grad_fitted.as_ref().map(|g| g.as_array()),
        grad_reml_score,
        grad_edf,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    let backward_for_t = if grad_reml_score != 0.0 {
        gaussian_reml_multi_closed_form_backward_from_fit(
            design.view(),
            y_view,
            penalty_view,
            weights_view,
            &fit,
            grad_lambda,
            grad_coefficients.as_ref().map(|g| g.as_array()),
            grad_fitted.as_ref().map(|g| g.as_array()),
            0.0,
            grad_edf,
        )
        .map_err(|err| py_value_error(err.to_string()))?
    } else {
        backward.clone()
    };
    let grad_x = &backward_for_t.grad_x;
    let mut grad_t = contract_input_loc_gradient(grad_x.view(), &jet)
        .map_err(|err| py_value_error(err.to_string()))?;
    if grad_reml_score != 0.0 {
        add_latent_outer_reml_score_gradient(
            &mut grad_t,
            grad_reml_score,
            design.view(),
            y_view,
            t_mat.view(),
            &jet,
            penalty_view,
            weights_view,
            &fit,
            sigma_eff_mode,
        )
        .map_err(py_value_error)?;
    }
    // Identifiability-mode additive contributions to grad_t plus log-normalizer
    // adjoints. Fixes audit-revised claim that REML ARD/AuxPrior selection
    // needs the normalized prior terms, not only raw quadratic gradients.
    let mut grad_aux_log_strength: Option<f64> = None;
    let mut grad_dim_selection_log_precision: Option<Array1<f64>> = None;
    if let Some(u_arr) = aux_u.as_ref() {
        let u_view = u_arr.as_array();
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, family, aux_strength)
            .map_err(py_value_error)?;
        let residual = &t_mat - &stats.targets;
        let projected_residual =
            aux_prior_targets(residual.view(), u_view, family).map_err(py_value_error)?;
        let grad_base = residual - projected_residual;
        for n in 0..n_obs {
            for a in 0..latent_dim {
                if grad_reml_score != 0.0 {
                    grad_t[n * latent_dim + a] +=
                        grad_reml_score * stats.strength.mu * grad_base[[n, a]];
                }
            }
        }
        grad_aux_log_strength = Some(
            grad_reml_score * (0.5 * stats.strength.mu * stats.residual_sq - 0.5 * n_obs as f64),
        );
    }
    if let Some(log_prec) = dim_selection_log_precision.as_ref() {
        let lp = log_prec.as_array();
        if lp.len() != latent_dim {
            return Err(py_value_error(format!(
                "dim_selection_log_precision length {} must equal latent_dim {}",
                lp.len(),
                latent_dim
            )));
        }
        let mut grad_log_prec = Array1::<f64>::zeros(latent_dim);
        for n in 0..n_obs {
            for a in 0..latent_dim {
                let prec = lp[a].exp();
                if grad_reml_score != 0.0 {
                    grad_t[n * latent_dim + a] += grad_reml_score * prec * t_mat[[n, a]];
                }
            }
        }
        for a in 0..latent_dim {
            let log_alpha = lp[a];
            let prec = log_alpha.exp();
            if !(prec.is_finite() && prec > 0.0) {
                return Err(py_value_error(format!(
                    "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                )));
            }
            let mut sq = 0.0_f64;
            for n in 0..n_obs {
                let v = t_mat[[n, a]];
                sq += v * v;
            }
            grad_log_prec[a] = grad_reml_score * (0.5 * prec * sq - 0.5 * (n_obs as f64));
        }
        grad_dim_selection_log_precision = Some(grad_log_prec);
    }
    let mut grad_t_matrix = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            grad_t_matrix[[n, a]] = grad_t[n * latent_dim + a];
        }
    }
    let out = PyDict::new(py);
    out.set_item("grad_t", grad_t_matrix.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", backward.grad_penalty.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    if let Some(grad) = grad_aux_log_strength {
        out.set_item("grad_aux_log_strength", grad)?;
        out.set_item("grad_log_mu", grad)?;
    } else {
        out.set_item("grad_aux_log_strength", py.None())?;
        out.set_item("grad_log_mu", py.None())?;
    }
    if let Some(grad) = grad_dim_selection_log_precision {
        out.set_item("grad_dim_selection_log_precision", grad.into_pyarray(py))?;
    } else {
        out.set_item("grad_dim_selection_log_precision", py.None())?;
    }
    Ok(out.unbind())
}

fn latent_glm_family_from_str(
    value: &str,
    tweedie_p: f64,
    negbin_theta: f64,
    beta_phi: f64,
) -> Result<LikelihoodSpec, String> {
    match value.to_ascii_lowercase().replace('_', "-").as_str() {
        "gaussian" | "gaussian-identity" => Ok(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(LinkFunction::Identity),
        )),
        "binomial" | "binomial-logit" | "logistic" => Ok(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(LinkFunction::Logit),
        )),
        "binomial-probit" | "probit" => Ok(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(LinkFunction::Probit),
        )),
        "binomial-cloglog" | "cloglog" => Ok(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(LinkFunction::CLogLog),
        )),
        "poisson" | "poisson-log" => Ok(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(LinkFunction::Log),
        )),
        "tweedie" | "tweedie-log" => {
            if !tweedie_p.is_finite() {
                return Err(format!("tweedie_p must be finite; got {tweedie_p}"));
            }
            Ok(LikelihoodSpec::new(
                ResponseFamily::Tweedie { p: tweedie_p },
                InverseLink::Standard(LinkFunction::Log),
            ))
        }
        "negbin" | "negbin-log" | "negative-binomial" | "negative-binomial-log" => {
            if !(negbin_theta.is_finite() && negbin_theta > 0.0) {
                return Err(format!(
                    "negbin_theta must be finite and > 0; got {negbin_theta}"
                ));
            }
            Ok(LikelihoodSpec::new(
                ResponseFamily::NegativeBinomial {
                    theta: negbin_theta,
                },
                InverseLink::Standard(LinkFunction::Log),
            ))
        }
        "beta" | "beta-logit" | "beta-regression" | "beta-regression-logit" => {
            if !(beta_phi.is_finite() && beta_phi > 0.0) {
                return Err(format!("beta_phi must be finite and > 0; got {beta_phi}"));
            }
            Ok(LikelihoodSpec::new(
                ResponseFamily::Beta { phi: beta_phi },
                InverseLink::Standard(LinkFunction::Logit),
            ))
        }
        "gamma" | "gamma-log" => Ok(LikelihoodSpec::new(
            ResponseFamily::Gamma,
            InverseLink::Standard(LinkFunction::Log),
        )),
        other => Err(format!(
            "unsupported latent GLM family {other:?}; supported families are gaussian-identity, binomial-logit, binomial-probit, binomial-cloglog, poisson-log, tweedie-log, negbin-log, beta-regression-logit, gamma-log"
        )),
    }
}

fn glm_reml_fit_latent_impl(
    t_flat: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    family: LikelihoodSpec,
    aux_u: Option<ArrayView2<'_, f64>>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
    dim_selection_precision: Option<ArrayView1<'_, f64>>,
    analytic_penalties: Option<&AnalyticPenaltyRegistry>,
) -> Result<
    (
        gam::estimate::ExternalOptimResult,
        Array2<f64>,
        Array2<f64>,
        Option<LatentAuxStrengthState>,
    ),
    String,
> {
    if y.ncols() != 1 {
        return Err(format!(
            "glm_reml_fit_latent requires y with one column; got {}",
            y.ncols()
        ));
    }
    let (design, t_mat) = build_latent_duchon_design(t_flat, n_obs, latent_dim, centers, m)?;
    if penalty.dim() != (design.ncols(), design.ncols()) {
        return Err(format!(
            "penalty shape mismatch: expected {}x{}, got {}x{}",
            design.ncols(),
            design.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    let y_vec = y.column(0).to_owned();
    let weights_owned = match weights {
        Some(w) => gaussian_reml_weight_vector_local(n_obs, Some(w))?,
        None => Array1::ones(n_obs),
    };
    let offset = Array1::<f64>::zeros(n_obs);
    let penalty_block = BlockwisePenalty::new(0..design.ncols(), penalty.to_owned());
    let opts = ExternalOptimOptions {
        family,
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        max_iter: 100,
        tol: 1e-7,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        penalty_shrinkage_floor: None,
        rho_prior: RhoPrior::Flat,
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let heuristic_lambda = init_lambda.map(|lambda| [lambda]);
    let mut fit = optimize_external_designwith_heuristic_lambdas(
        y_vec.view(),
        weights_owned.view(),
        design.clone(),
        offset.view(),
        vec![penalty_block],
        heuristic_lambda.as_ref().map(|values| values.as_slice()),
        &opts,
    )
    .map_err(|err| err.to_string())?;
    let mut latent_prior_score = 0.0_f64;
    let mut aux_strength_state = None;
    if let Some(u_view) = aux_u {
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, aux_family, aux_strength)?;
        latent_prior_score += stats.score;
        aux_strength_state = Some(stats.strength);
    }
    if let Some(log_prec) = dim_selection_precision {
        if log_prec.len() != latent_dim {
            return Err(format!(
                "dim_selection_log_precision length {} must equal latent_dim {}",
                log_prec.len(),
                latent_dim
            ));
        }
        for a in 0..latent_dim {
            let log_alpha = log_prec[a];
            let alpha = log_alpha.exp();
            if !(alpha.is_finite() && alpha > 0.0) {
                return Err(format!(
                    "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                ));
            }
            let mut sq = 0.0_f64;
            for n in 0..n_obs {
                let v = t_mat[[n, a]];
                sq += v * v;
            }
            latent_prior_score += 0.5 * alpha * sq - 0.5 * (n_obs as f64) * log_alpha;
        }
    }
    if let Some(registry) = analytic_penalties {
        latent_prior_score +=
            analytic_penalty_value_for_targets(registry, t_flat, Some(fit.beta.view()))?;
    }
    fit.reml_score += latent_prior_score;
    Ok((fit, design, t_mat, aux_strength_state))
}

fn set_ok_glm_latent_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    fit: gam::estimate::ExternalOptimResult,
    n_obs: usize,
    p: usize,
    aux_strength_state: Option<LatentAuxStrengthState>,
) -> PyResult<()> {
    let pirls = fit.artifacts.pirls.as_ref().ok_or_else(|| {
        py_value_error("latent GLM fit did not return PIRLS artifacts".to_string())
    })?;
    let lambda = fit.lambdas.get(0).copied().unwrap_or(f64::NAN);
    let rho = lambda.ln();
    let mut coefficients = Array2::<f64>::zeros((p, 1));
    for row in 0..p.min(fit.beta.len()) {
        coefficients[[row, 0]] = fit.beta[row];
    }
    let mut fitted = Array2::<f64>::zeros((n_obs, 1));
    for row in 0..n_obs.min(pirls.finalmu.len()) {
        fitted[[row, 0]] = pirls.finalmu[row];
    }
    out.set_item(
        "status",
        if fit.outer_converged {
            "ok"
        } else {
            "not_converged"
        },
    )?;
    out.set_item("lambda", lambda)?;
    out.set_item("rho", rho)?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("reml_grad_lambda", f64::NAN)?;
    out.set_item("reml_hess_lambda", f64::NAN)?;
    out.set_item("reml_grad_rho", f64::NAN)?;
    out.set_item("reml_hess_rho", f64::NAN)?;
    out.set_item("edf", pirls.edf)?;
    out.set_item("coefficients", coefficients.into_pyarray(py))?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item(
        "sigma2",
        Array1::from_vec(vec![fit.standard_deviation * fit.standard_deviation]).into_pyarray(py),
    )?;
    out.set_item(
        "cache_penalty_eigenvalues",
        Array1::<f64>::zeros(0).into_pyarray(py),
    )?;
    out.set_item(
        "cache_eigenvectors",
        Array2::<f64>::zeros((0, 0)).into_pyarray(py),
    )?;
    out.set_item(
        "cache_coefficient_basis",
        Array2::<f64>::zeros((0, 0)).into_pyarray(py),
    )?;
    out.set_item("cache_xtwx_fingerprint", 0_u64)?;
    out.set_item("cache_penalty_fingerprint", 0_u64)?;
    out.set_item("cache_logdet_xtwx", f64::NAN)?;
    out.set_item("cache_logdet_penalty_positive", f64::NAN)?;
    out.set_item("cache_penalty_rank", 0_usize)?;
    out.set_item("cache_nullity", 0_usize)?;
    set_aux_strength_items(py, out, aux_strength_state)?;
    Ok(())
}

#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    family,
    tweedie_p = 1.5,
    negbin_theta = 1.0,
    beta_phi = 1.0,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    analytic_penalties = None,
))]
fn glm_reml_fit_latent<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    family: String,
    tweedie_p: f64,
    negbin_theta: f64,
    beta_phi: f64,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    analytic_penalties: Option<String>,
) -> PyResult<Py<PyDict>> {
    let analytic_penalties: Option<serde_json::Value> = match analytic_penalties {
        Some(s) => Some(serde_json::from_str(&s).map_err(|e| py_value_error(e.to_string()))?),
        None => None,
    };
    let latent_payload = serde_json::json!({"t": {"name": "t", "n": n_obs, "d": latent_dim}});
    let registry = build_analytic_penalty_registry_from_json(
        Some(&latent_payload),
        analytic_penalties.as_ref(),
    )
    .map_err(py_value_error)?;
    let family_name = family.clone();
    let family_normalized = family_name.to_ascii_lowercase().replace('_', "-");
    let aux_family = match aux_family.to_ascii_lowercase().as_str() {
        "ridge" => AuxPriorFamily::Ridge,
        "linear" => AuxPriorFamily::Linear,
        other => {
            return Err(py_value_error(format!(
                "aux_family must be 'ridge' or 'linear'; got {other:?}"
            )));
        }
    };
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let centers_values = centers.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let fisher_values = fisher_w.as_ref().map(|w| w.as_array().to_owned());
    let aux_u_values = aux_u.as_ref().map(|a| a.as_array().to_owned());
    let dim_selection_values = dim_selection_log_precision
        .as_ref()
        .map(|a| a.as_array().to_owned());
    if y_values.ncols() > 1
        || matches!(
            family_normalized.as_str(),
            "multinomial" | "multinomial-logit" | "softmax" | "categorical-logit"
        )
    {
        let (design, t_mat) = build_latent_duchon_design(
            t_values.view(),
            n_obs,
            latent_dim,
            centers_values.view(),
            m,
        )
        .map_err(py_value_error)?;
        let (prior_score, aux_strength_state) = latent_prior_score_and_aux_state_for_t(
            t_mat.view(),
            aux_u_values.as_ref().map(|a| a.view()),
            aux_family,
            aux_strength,
            dim_selection_values.as_ref().map(|a| a.view()),
        )
        .map_err(py_value_error)?;
        let analytic_score = analytic_penalty_value_for_targets(&registry, t_values.view(), None)
            .map_err(py_value_error)?;
        return dense_fisher_glm_fit_to_pydict(
            py,
            design.view(),
            y_values.view(),
            penalty_values.view(),
            weight_values.as_ref().map(|w| w.view()),
            fisher_values.as_ref().map(|w| w.view()),
            init_lambda,
            &family_name,
            prior_score + analytic_score,
            aux_strength_state,
        );
    }
    let family = latent_glm_family_from_str(&family, tweedie_p, negbin_theta, beta_phi)
        .map_err(py_value_error)?;
    let analytic_penalties_for_thread = analytic_penalties.clone();
    let latent_payload_for_thread = latent_payload.clone();
    let (fit, design, _, aux_strength_state) =
        detach_py_result(py, "glm_reml_fit_latent", move || {
            let registry = build_analytic_penalty_registry_from_json(
                Some(&latent_payload_for_thread),
                analytic_penalties_for_thread.as_ref(),
            )?;
            let effective_weights = latent_scalar_weights_with_fisher(
                n_obs,
                weight_values.as_ref().map(|w| w.view()),
                fisher_values.as_ref().map(|w| w.view()),
            )?;
            glm_reml_fit_latent_impl(
                t_values.view(),
                y_values.view(),
                n_obs,
                latent_dim,
                centers_values.view(),
                m,
                penalty_values.view(),
                effective_weights.as_ref().map(|w| w.view()),
                init_lambda,
                family,
                aux_u_values.as_ref().map(|a| a.view()),
                aux_family,
                aux_strength,
                dim_selection_values.as_ref().map(|a| a.view()),
                Some(&registry),
            )
        })?;
    let out = PyDict::new(py);
    set_ok_glm_latent_items(py, &out, fit, n_obs, design.ncols(), aux_strength_state)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    family,
    grad_reml_score = 1.0,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    tweedie_p = 1.5,
    negbin_theta = 1.0,
    beta_phi = 1.0,
    basis_kind = "duchon".to_string(),
))]
fn glm_reml_fit_latent_backward<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    family: String,
    grad_reml_score: f64,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    tweedie_p: f64,
    negbin_theta: f64,
    beta_phi: f64,
    basis_kind: String,
) -> PyResult<Py<PyDict>> {
    let family = latent_glm_family_from_str(&family, tweedie_p, negbin_theta, beta_phi)
        .map_err(py_value_error)?;
    let aux_family = match aux_family.to_ascii_lowercase().as_str() {
        "ridge" => AuxPriorFamily::Ridge,
        "linear" => AuxPriorFamily::Linear,
        other => {
            return Err(py_value_error(format!(
                "aux_family must be 'ridge' or 'linear'; got {other:?}"
            )));
        }
    };
    let basis_kind_normalized = latent_basis_kind(&basis_kind).map_err(py_value_error)?;
    if basis_kind_normalized != "duchon" {
        return Err(PyNotImplementedError::new_err(format!(
            "glm_reml_fit_latent_backward currently builds only Duchon latent designs; derivative hook exists for {basis_kind_normalized:?}"
        )));
    }
    let effective_weights = latent_scalar_weights_with_fisher(
        n_obs,
        weights.as_ref().map(|w| w.as_array()),
        fisher_w.as_ref().map(|w| w.as_array()),
    )
    .map_err(py_value_error)?;
    let (fit, design, t_mat, _) = glm_reml_fit_latent_impl(
        t.as_array(),
        y.as_array(),
        n_obs,
        latent_dim,
        centers.as_array(),
        m,
        penalty.as_array(),
        effective_weights.as_ref().map(|w| w.view()),
        init_lambda,
        family,
        aux_u.as_ref().map(|a| a.as_array()),
        aux_family,
        aux_strength,
        dim_selection_log_precision.as_ref().map(|a| a.as_array()),
        None,
    )
    .map_err(py_value_error)?;
    let pirls = fit.artifacts.pirls.as_ref().ok_or_else(|| {
        py_value_error("latent GLM fit did not return PIRLS artifacts".to_string())
    })?;
    let h = pirls
        .dense_stabilizedhessian_transformed("latent GLM backward")
        .map_err(|err| py_value_error(err.to_string()))?;
    let factor = factorize_symmetricwith_fallback(
        gam::faer_ndarray::FaerArrayView::new(&h).as_ref(),
        Side::Lower,
    )
    .map_err(|err| py_value_error(format!("latent GLM Hessian factorization failed: {err}")))?;
    let qs = &pirls.reparam_result.qs;
    let beta_t = pirls.beta_transformed.as_ref();
    let q = beta_t.len();
    let p = design.ncols();
    let jet = latent_input_location_jet(
        basis_kind_normalized,
        t_mat.view(),
        centers.as_array(),
        m,
        None,
        None,
        None,
    )
    .map_err(py_value_error)?;
    if jet.shape()[0] != n_obs || jet.shape()[1] != p || jet.shape()[2] != latent_dim {
        return Err(py_value_error(format!(
            "latent input-location jet shape mismatch: expected {}x{}x{}, got {}x{}x{}",
            n_obs,
            p,
            latent_dim,
            jet.shape()[0],
            jet.shape()[1],
            jet.shape()[2],
        )));
    }
    let mut grad_t = Array1::<f64>::zeros(n_obs * latent_dim);
    let mut rhs = Array2::<f64>::zeros((q, 1));
    let mut direction_t = Array1::<f64>::zeros(q);
    let mut direction_orig = Array1::<f64>::zeros(p);
    for n in 0..n_obs {
        for col in 0..q {
            let mut value = 0.0_f64;
            for k in 0..p {
                value += design[[n, k]] * qs[[k, col]];
            }
            rhs[[col, 0]] = value;
        }
        {
            let mut rhs_view = array2_to_matmut(&mut rhs);
            factor.solve_in_place(rhs_view.as_mut());
        }
        let score_eta =
            pirls.solveweights[n] * (pirls.final_eta[n] - pirls.solveworking_response[n]);
        for col in 0..q {
            direction_t[col] = score_eta * beta_t[col] + pirls.finalweights[n] * rhs[[col, 0]];
        }
        direction_orig.fill(0.0);
        for k in 0..p {
            let mut value = 0.0_f64;
            for col in 0..q {
                value += qs[[k, col]] * direction_t[col];
            }
            direction_orig[k] = value;
        }
        for a in 0..latent_dim {
            let mut acc = 0.0_f64;
            for k in 0..p {
                acc += direction_orig[k] * jet[[n, k, a]];
            }
            grad_t[n * latent_dim + a] += grad_reml_score * acc;
        }
    }

    let mut grad_aux_log_strength: Option<f64> = None;
    if let Some(u_arr) = aux_u.as_ref() {
        let u_view = u_arr.as_array();
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, aux_family, aux_strength)
            .map_err(py_value_error)?;
        let residual = &t_mat - &stats.targets;
        let projected_residual =
            aux_prior_targets(residual.view(), u_view, aux_family).map_err(py_value_error)?;
        let grad_base = residual - projected_residual;
        for n in 0..n_obs {
            for a in 0..latent_dim {
                grad_t[n * latent_dim + a] +=
                    grad_reml_score * stats.strength.mu * grad_base[[n, a]];
            }
        }
        grad_aux_log_strength = Some(
            grad_reml_score * (0.5 * stats.strength.mu * stats.residual_sq - 0.5 * n_obs as f64),
        );
    }
    if let Some(log_prec) = dim_selection_log_precision.as_ref() {
        let lp = log_prec.as_array();
        if lp.len() != latent_dim {
            return Err(py_value_error(format!(
                "dim_selection_log_precision length {} must equal latent_dim {}",
                lp.len(),
                latent_dim
            )));
        }
        for n in 0..n_obs {
            for a in 0..latent_dim {
                let prec = lp[a].exp();
                if !(prec.is_finite() && prec > 0.0) {
                    return Err(py_value_error(format!(
                        "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                    )));
                }
                grad_t[n * latent_dim + a] += grad_reml_score * prec * t_mat[[n, a]];
            }
        }
    }
    let mut grad_t_matrix = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            grad_t_matrix[[n, a]] = grad_t[n * latent_dim + a];
        }
    }
    let out = PyDict::new(py);
    out.set_item("grad_t", grad_t_matrix.into_pyarray(py))?;
    if let Some(grad) = grad_aux_log_strength {
        out.set_item("grad_aux_log_strength", grad)?;
        out.set_item("grad_log_mu", grad)?;
    } else {
        out.set_item("grad_aux_log_strength", py.None())?;
        out.set_item("grad_log_mu", py.None())?;
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
    let status = if fit.lambda.is_finite() && fit.reml_score.is_finite() {
        "ok"
    } else {
        "diverged"
    };
    out.set_item("status", status)?;
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

fn batched_gaussian_reml_fits_from_pydict(
    state: &Bound<'_, PyDict>,
    row_offsets: ArrayView1<'_, usize>,
) -> Result<Vec<Option<gam::gaussian_reml::GaussianRemlMultiResult>>, String> {
    fn get<'py>(state: &'py Bound<'py, PyDict>, key: &str) -> Result<Bound<'py, PyAny>, String> {
        state
            .get_item(key)
            .map_err(|err| err.to_string())?
            .ok_or_else(|| format!("forward_state is missing key {key:?}"))
    }
    if row_offsets.len() < 2 {
        return Err("row_offsets must contain at least [0, n]".to_string());
    }
    let batch = row_offsets.len() - 1;

    let statuses = get(state, "status")?
        .extract::<Vec<String>>()
        .map_err(|err| err.to_string())?;
    if statuses.len() != batch {
        return Err(format!(
            "forward_state[\"status\"] length mismatch: expected {batch}, got {}",
            statuses.len()
        ));
    }
    let lambdas = get(state, "lambda")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let lambdas = lambdas.as_array();
    let rhos = get(state, "rho")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let rhos = rhos.as_array();
    let reml_scores = get(state, "reml_score")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_scores = reml_scores.as_array();
    let reml_grad_lambdas = get(state, "reml_grad_lambda")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_grad_lambdas = reml_grad_lambdas.as_array();
    let reml_hess_lambdas = get(state, "reml_hess_lambda")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_hess_lambdas = reml_hess_lambdas.as_array();
    let reml_grad_rhos = get(state, "reml_grad_rho")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_grad_rhos = reml_grad_rhos.as_array();
    let reml_hess_rhos = get(state, "reml_hess_rho")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_hess_rhos = reml_hess_rhos.as_array();
    let edf = get(state, "edf")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let edf = edf.as_array();
    let coefficients = get(state, "coefficients")?
        .extract::<PyReadonlyArray3<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let coefficients = coefficients.as_array();
    let fitted = get(state, "fitted")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let fitted = fitted.as_array();
    let sigma2 = get(state, "sigma2")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let sigma2 = sigma2.as_array();
    let cache_penalty_eigenvalues = get(state, "cache_penalty_eigenvalues")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_penalty_eigenvalues = cache_penalty_eigenvalues.as_array();
    let cache_eigenvectors = get(state, "cache_eigenvectors")?
        .extract::<PyReadonlyArray3<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_eigenvectors = cache_eigenvectors.as_array();
    let cache_coefficient_basis = get(state, "cache_coefficient_basis")?
        .extract::<PyReadonlyArray3<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_coefficient_basis = cache_coefficient_basis.as_array();
    let cache_xtwx_fingerprints = get(state, "cache_xtwx_fingerprints")?
        .extract::<PyReadonlyArray1<'_, u64>>()
        .map_err(|err| err.to_string())?;
    let cache_xtwx_fingerprints = cache_xtwx_fingerprints.as_array();
    let cache_penalty_fingerprints = get(state, "cache_penalty_fingerprints")?
        .extract::<PyReadonlyArray1<'_, u64>>()
        .map_err(|err| err.to_string())?;
    let cache_penalty_fingerprints = cache_penalty_fingerprints.as_array();
    let cache_logdet_xtwx = get(state, "cache_logdet_xtwx")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_logdet_xtwx = cache_logdet_xtwx.as_array();
    let cache_logdet_penalty_positive = get(state, "cache_logdet_penalty_positive")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_logdet_penalty_positive = cache_logdet_penalty_positive.as_array();
    let cache_penalty_ranks = get(state, "cache_penalty_ranks")?
        .extract::<PyReadonlyArray1<'_, i64>>()
        .map_err(|err| err.to_string())?;
    let cache_penalty_ranks = cache_penalty_ranks.as_array();
    let cache_nullities = get(state, "cache_nullities")?
        .extract::<PyReadonlyArray1<'_, i64>>()
        .map_err(|err| err.to_string())?;
    let cache_nullities = cache_nullities.as_array();

    let mut fits = Vec::with_capacity(batch);
    for b in 0..batch {
        if statuses[b] != "ok" {
            fits.push(None);
            continue;
        }
        let rank = cache_penalty_ranks[b];
        let nullity = cache_nullities[b];
        if rank < 0 || nullity < 0 {
            return Err(format!(
                "forward_state cache_penalty_ranks[{b}]={rank} or cache_nullities[{b}]={nullity} must be non-negative"
            ));
        }
        let start = row_offsets[b];
        let end = row_offsets[b + 1];
        if start > end || end > fitted.nrows() {
            return Err(format!(
                "row_offsets[{b}..{}]=({start},{end}) outside fitted shape {}",
                b + 1,
                fitted.nrows()
            ));
        }
        fits.push(Some(gam::gaussian_reml::GaussianRemlMultiResult {
            lambda: lambdas[b],
            rho: rhos[b],
            coefficients: coefficients.slice(s![b, .., ..]).to_owned(),
            fitted: fitted.slice(s![start..end, ..]).to_owned(),
            reml_score: reml_scores[b],
            reml_grad_lambda: reml_grad_lambdas[b],
            reml_hess_lambda: reml_hess_lambdas[b],
            reml_grad_rho: reml_grad_rhos[b],
            reml_hess_rho: reml_hess_rhos[b],
            edf: edf[b],
            sigma2: sigma2.slice(s![b, ..]).to_owned(),
            cache: gam::gaussian_reml::GaussianRemlEigenCache {
                penalty_eigenvalues: cache_penalty_eigenvalues.slice(s![b, ..]).to_owned(),
                eigenvectors: cache_eigenvectors.slice(s![b, .., ..]).to_owned(),
                coefficient_basis: cache_coefficient_basis.slice(s![b, .., ..]).to_owned(),
                xtwx_fingerprint: cache_xtwx_fingerprints[b],
                penalty_fingerprint: cache_penalty_fingerprints[b],
                logdet_xtwx: cache_logdet_xtwx[b],
                logdet_penalty_positive: cache_logdet_penalty_positive[b],
                penalty_rank: rank as usize,
                nullity: nullity as usize,
            },
        }));
    }
    Ok(fits)
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
    // Per-fit cache stacked across the batch — populated for fits whose status
    // is "ok". Backward callers re-bind these into `GaussianRemlMultiResult`
    // and route to `_from_fit`, skipping the redundant ρ-search + cache build
    // that would otherwise double per-step cost at training-loop frequency.
    cache_penalty_eigenvalues: Array2<f64>,
    cache_eigenvectors: Array3<f64>,
    cache_coefficient_basis: Array3<f64>,
    cache_xtwx_fingerprints: Array1<u64>,
    cache_penalty_fingerprints: Array1<u64>,
    cache_logdet_xtwx: Array1<f64>,
    cache_logdet_penalty_positive: Array1<f64>,
    cache_penalty_ranks: Array1<i64>,
    cache_nullities: Array1<i64>,
}

struct BatchedGaussianRemlBackwardResult {
    statuses: Vec<String>,
    grad_x: Array2<f64>,
    grad_y: Array2<f64>,
    grad_penalty: Array2<f64>,
    grad_weights: Array1<f64>,
}

struct PositionGaussianRemlBackwardResult {
    grad_t: Array1<f64>,
    grad_y: Array2<f64>,
    grad_penalty: Array2<f64>,
    grad_weights: Array1<f64>,
    grad_by: Option<Array1<f64>>,
}

struct BatchedPositionGaussianRemlBackwardResult {
    statuses: Vec<String>,
    grad_t: Array1<f64>,
    grad_y: Array2<f64>,
    grad_penalty: Array2<f64>,
    grad_weights: Array1<f64>,
    grad_by: Option<Array1<f64>>,
}

fn gaussian_reml_fit_formula_table_impl(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    y: ArrayView2<'_, f64>,
    config_json: Option<&str>,
    fisher_rao_w: Option<ArrayView3<'_, f64>>,
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
    if !standard.family.is_gaussian_identity() {
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
    let design = gam::smooth::build_term_collection_design(standard.data.view(), &standard.spec)
        .map_err(|err| format!("failed to build formula design matrix: {err}"))?;
    let x = design
        .design
        .try_to_dense_by_chunks("closed_form_gaussian_reml_formula design")?;
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
    if let Some(w) = fisher_rao_w {
        return gaussian_reml_formula_table_dense_fisher(
            x.view(),
            y,
            penalty.view(),
            standard.weights.view(),
            w,
        );
    }
    gam::gaussian_reml::gaussian_reml_multi_closed_form(
        x.view(),
        y,
        penalty.view(),
        Some(standard.weights.view()),
        None,
    )
    .map_err(|err| err.to_string())
}

fn gaussian_reml_formula_table_dense_fisher(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    fisher_w: ArrayView3<'_, f64>,
) -> Result<gam::gaussian_reml::GaussianRemlMultiResult, String> {
    let n = x.nrows();
    let k = x.ncols();
    let p_out = y.ncols();
    if y.nrows() != n {
        return Err(format!(
            "dense Fisher formula response row mismatch: formula design has {n} rows but Y has {}",
            y.nrows()
        ));
    }
    validate_dense_fisher_w(n, p_out, fisher_w)?;
    if weights.len() != n {
        return Err(format!(
            "dense Fisher formula weight length mismatch: expected {n}, got {}",
            weights.len()
        ));
    }
    if weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("dense Fisher formula weights must be finite and non-negative".to_string());
    }

    let mut x_weighted = Array2::<f64>::zeros((n * p_out, k * p_out));
    let mut y_weighted = Array2::<f64>::zeros((n * p_out, 1));
    for row in 0..n {
        let mut block = fisher_w.slice(s![row, .., ..]).to_owned();
        for a in 0..p_out {
            for b in (a + 1)..p_out {
                let avg = 0.5 * (block[[a, b]] + block[[b, a]]);
                block[[a, b]] = avg;
                block[[b, a]] = avg;
            }
        }
        let lower = block
            .cholesky(Side::Lower)
            .map_err(|err| {
                format!("fisher_rao_w row {row} must be positive-definite for dense REML: {err}")
            })?
            .lower_triangular();
        let scale = weights[row].sqrt();
        for metric_axis in 0..p_out {
            let stacked_row = row * p_out + metric_axis;
            let mut y_value = 0.0;
            for output in 0..p_out {
                let l_t = scale * lower[[output, metric_axis]];
                y_value += l_t * y[[row, output]];
                let col_offset = output * k;
                for col in 0..k {
                    x_weighted[[stacked_row, col_offset + col]] = l_t * x[[row, col]];
                }
            }
            y_weighted[[stacked_row, 0]] = y_value;
        }
    }

    let mut block_penalty = Array2::<f64>::zeros((k * p_out, k * p_out));
    for output in 0..p_out {
        let offset = output * k;
        for row in 0..k {
            for col in 0..k {
                block_penalty[[offset + row, offset + col]] = penalty[[row, col]];
            }
        }
    }
    let mut fit = gam::gaussian_reml::gaussian_reml_multi_closed_form(
        x_weighted.view(),
        y_weighted.view(),
        block_penalty.view(),
        None,
        None,
    )
    .map_err(|err| err.to_string())?;

    let mut coefficients = Array2::<f64>::zeros((k, p_out));
    for output in 0..p_out {
        for col in 0..k {
            coefficients[[col, output]] = fit.coefficients[[output * k + col, 0]];
        }
    }
    let fitted = x.dot(&coefficients);
    let denom = n.saturating_sub(k).max(1) as f64;
    let mut sigma2 = Array1::<f64>::zeros(p_out);
    for row in 0..n {
        for output in 0..p_out {
            let resid = y[[row, output]] - fitted[[row, output]];
            sigma2[output] += weights[row] * resid * resid;
        }
    }
    sigma2.mapv_inplace(|v| v / denom);
    fit.coefficients = coefficients;
    fit.fitted = fitted;
    fit.sigma2 = sigma2;
    Ok(fit)
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

    // Phase A: compute X'WX per fit in parallel (CPU or per-fit GPU dispatch
    // via `fast_xt_diag_x`). Per-fit X'WX cost is `O(n_b · p²)`; this phase
    // produces K p×p matrices that feed the batched Cholesky in Phase B.
    let xtwx_phase: Vec<Option<Array2<f64>>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return None;
            }
            let x_slice = x.slice(s![start..end, ..]);
            let owned_weight: Array1<f64> = match weights.as_ref() {
                Some(w) => w.slice(s![start..end]).to_owned(),
                None => Array1::ones(end - start),
            };
            Some(gam::faer_ndarray::fast_xt_diag_x(&x_slice, &owned_weight))
        })
        .collect();

    // Phase B: assemble live X'WX matrices and run the batched cache build.
    // The Cholesky step inside collapses to a single `cusolverDnDpotrfBatched`
    // call when policy + uniform shape allow; otherwise falls back to
    // per-fit Cholesky inside the helper. The remaining whitened-penalty
    // eigh stays per-fit (cuSOLVER has no batched symmetric eigensolver).
    let mut live_indices: Vec<usize> = Vec::with_capacity(batch);
    let mut live_xtwx: Vec<Array2<f64>> = Vec::with_capacity(batch);
    for (b, slot) in xtwx_phase.into_iter().enumerate() {
        if let Some(xtwx) = slot {
            live_indices.push(b);
            live_xtwx.push(xtwx);
        }
    }
    let batched_caches = build_gaussian_reml_eigen_cache_batched(live_xtwx, penalty, None);
    let mut prebuilt_caches: Vec<Option<gam::gaussian_reml::GaussianRemlEigenCache>> =
        (0..batch).map(|_| None).collect();
    for (i, cache_result) in batched_caches.into_iter().enumerate() {
        if let Ok(cache) = cache_result {
            prebuilt_caches[live_indices[i]] = Some(cache);
        }
    }

    // Phase C: per-fit completion. Each fit either uses the prebuilt cache
    // (skipping its chol + eigh in `prepare_gaussian_reml`) or falls through
    // to a fresh build when the batched cache build dropped that element.
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
            let cache_ref = prebuilt_caches[b].as_ref();
            match gaussian_reml_multi_closed_form_with_cache(
                x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty,
                weight_slice,
                init_lambda,
                cache_ref,
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
    let mut cache_penalty_eigenvalues = Array2::<f64>::zeros((batch, p));
    let mut cache_eigenvectors = Array3::<f64>::zeros((batch, p, p));
    let mut cache_coefficient_basis = Array3::<f64>::zeros((batch, p, p));
    let mut cache_xtwx_fingerprints = Array1::<u64>::zeros(batch);
    let mut cache_penalty_fingerprints = Array1::<u64>::zeros(batch);
    let mut cache_logdet_xtwx = Array1::<f64>::zeros(batch);
    let mut cache_logdet_penalty_positive = Array1::<f64>::zeros(batch);
    let mut cache_penalty_ranks = Array1::<i64>::zeros(batch);
    let mut cache_nullities = Array1::<i64>::zeros(batch);

    for result in fit_results {
        let (b, fit) = result?;
        if let Some(fit) = fit {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = if fit.lambda.is_finite() && fit.reml_score.is_finite() {
                "ok".to_string()
            } else {
                "diverged".to_string()
            };
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
            cache_penalty_eigenvalues
                .slice_mut(s![b, ..])
                .assign(&fit.cache.penalty_eigenvalues);
            cache_eigenvectors
                .slice_mut(s![b, .., ..])
                .assign(&fit.cache.eigenvectors);
            cache_coefficient_basis
                .slice_mut(s![b, .., ..])
                .assign(&fit.cache.coefficient_basis);
            cache_xtwx_fingerprints[b] = fit.cache.xtwx_fingerprint;
            cache_penalty_fingerprints[b] = fit.cache.penalty_fingerprint;
            cache_logdet_xtwx[b] = fit.cache.logdet_xtwx;
            cache_logdet_penalty_positive[b] = fit.cache.logdet_penalty_positive;
            cache_penalty_ranks[b] = fit.cache.penalty_rank as i64;
            cache_nullities[b] = fit.cache.nullity as i64;
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
        cache_penalty_eigenvalues,
        cache_eigenvectors,
        cache_coefficient_basis,
        cache_xtwx_fingerprints,
        cache_penalty_fingerprints,
        cache_logdet_xtwx,
        cache_logdet_penalty_positive,
        cache_penalty_ranks,
        cache_nullities,
    })
}

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
    grad_edf: Option<ArrayView1<'_, f64>>,
    forward_fits: Option<&[Option<gam::gaussian_reml::GaussianRemlMultiResult>]>,
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
        grad_edf,
    )?;
    if let Some(fits) = forward_fits {
        if fits.len() != batch {
            return Err(format!(
                "forward_state fit count mismatch: expected {batch}, got {}",
                fits.len()
            ));
        }
    }

    // When the caller supplied forward state, every active fit gets its
    // own `GaussianRemlMultiBackwardProblem` and the K-way batched entry
    // point handles the K-aggregate inverse-Hessian computation via
    // cuBLAS strided-batched gemm (one device call instead of K). Without
    // state, each fit refits internally; that path keeps the existing
    // per-fit par_iter shape.
    let results: Vec<
        Result<
            (
                usize,
                Option<gam::gaussian_reml::GaussianRemlBackwardResult>,
            ),
            String,
        >,
    > = if let Some(fits) = forward_fits {
        let mut active: Vec<(usize, GaussianRemlMultiBackwardProblem<'_>)> =
            Vec::with_capacity(batch);
        for b in 0..batch {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                continue;
            }
            let Some(fit) = fits[b].as_ref() else {
                continue;
            };
            active.push((
                b,
                GaussianRemlMultiBackwardProblem {
                    x: x.slice(s![start..end, ..]),
                    y: y.slice(s![start..end, ..]),
                    weights: weights.as_ref().map(|w| w.slice(s![start..end])),
                    fit,
                    grad_lambda: grad_lambda.as_ref().map_or(0.0, |g| g[b]),
                    grad_coefficients: grad_coefficients.as_ref().map(|g| g.slice(s![b, .., ..])),
                    grad_fitted: grad_fitted.as_ref().map(|g| g.slice(s![start..end, ..])),
                    grad_reml_score: grad_reml_score.as_ref().map_or(0.0, |g| g[b]),
                    grad_edf: grad_edf.as_ref().map_or(0.0, |g| g[b]),
                },
            ));
        }
        let (indices, problems): (Vec<usize>, Vec<GaussianRemlMultiBackwardProblem<'_>>) =
            active.into_iter().unzip();
        let batch_results = gaussian_reml_multi_closed_form_backward_batch(&problems, penalty);
        batch_results
            .into_iter()
            .zip(indices.into_iter())
            .map(|(result, b)| match result {
                Ok(backward) => Ok((b, Some(backward))),
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!("batched Gaussian REML backward {b} failed: {err}")),
            })
            .collect()
    } else {
        (0..batch)
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
                let upstream_edf = grad_edf.as_ref().map_or(0.0, |g| g[b]);
                let upstream_coefficients =
                    grad_coefficients.as_ref().map(|g| g.slice(s![b, .., ..]));
                let upstream_fitted = grad_fitted.as_ref().map(|g| g.slice(s![start..end, ..]));
                let x_slice = x.slice(s![start..end, ..]);
                let y_slice = y.slice(s![start..end, ..]);
                let backward_result = gaussian_reml_multi_closed_form_backward(
                    x_slice,
                    y_slice,
                    penalty,
                    weight_slice,
                    init_lambda,
                    upstream_lambda,
                    upstream_coefficients,
                    upstream_fitted,
                    upstream_reml_score,
                    upstream_edf,
                );
                match backward_result {
                    Ok(backward) => Ok((b, Some(backward))),
                    Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                    Err(err) => Err(format!("batched Gaussian REML backward {b} failed: {err}")),
                }
            })
            .collect()
    };

    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut grad_x = Array2::<f64>::zeros(x.dim());
    let mut grad_y = Array2::<f64>::zeros(y.dim());
    let mut grad_penalty = Array2::<f64>::zeros(penalty.dim());
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
            grad_penalty += &backward.grad_penalty;
            grad_weights
                .slice_mut(s![start..end])
                .assign(&backward.grad_weights);
        }
    }

    Ok(BatchedGaussianRemlBackwardResult {
        statuses,
        grad_x,
        grad_y,
        grad_penalty,
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
    grad_edf: Option<ArrayView1<'_, f64>>,
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
    if let Some(grad_edf) = grad_edf {
        if grad_edf.len() != batch {
            return Err(format!(
                "grad_edf length mismatch: expected {batch}, got {}",
                grad_edf.len()
            ));
        }
        if grad_edf.iter().any(|value| !value.is_finite()) {
            return Err("grad_edf must contain only finite values".to_string());
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
    grad_edf: f64,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
    forward_fit: Option<&gam::gaussian_reml::GaussianRemlMultiResult>,
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
    let backward = if let Some(fit) = forward_fit {
        gaussian_reml_multi_closed_form_backward_from_fit(
            fit_x,
            y,
            penalty,
            weights,
            fit,
            grad_lambda,
            grad_coefficients,
            grad_fitted,
            grad_reml_score,
            grad_edf,
        )
    } else {
        gaussian_reml_multi_closed_form_backward(
            fit_x,
            y,
            penalty,
            weights,
            init_lambda,
            grad_lambda,
            grad_coefficients,
            grad_fitted,
            grad_reml_score,
            grad_edf,
        )
    }
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
        grad_penalty: backward.grad_penalty,
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
    gaussian_reml_fit_positions_batched_streaming_impl(
        t,
        y,
        row_offsets,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
        penalty,
        weights,
        init_lambda,
        by,
        by_start_col,
    )
}

fn validate_position_batched_reml_common(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    knots_or_centers: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
) -> Result<(usize, usize, usize), String> {
    validate_vector("t", t)?;
    validate_vector("knots_or_centers", knots_or_centers)?;
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
    if y.ncols() == 0 {
        return Err("batched Gaussian REML requires non-empty Y columns".to_string());
    }
    if penalty.nrows() == 0 || penalty.ncols() != penalty.nrows() {
        return Err(format!(
            "penalty must be a non-empty square matrix; got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if by_start_col > penalty.nrows() {
        return Err(format!(
            "by_start_col must be <= number of columns; got {by_start_col} for {} columns",
            penalty.nrows()
        ));
    }
    if y.iter()
        .chain(penalty.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched Gaussian REML inputs must be finite".to_string());
    }
    if let Some(weights) = weights {
        if weights.len() != t.len() {
            return Err(format!(
                "weights length mismatch: expected {}, got {}",
                t.len(),
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
    if let Some(lambda) = init_lambda {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(format!(
                "init_lambda must be finite and positive when provided; got {lambda}"
            ));
        }
    }
    if let Some(by) = by {
        if by.len() != t.len() {
            return Err(format!(
                "by gate length mismatch: expected {}, got {}",
                t.len(),
                by.len()
            ));
        }
        if by.iter().any(|value| !value.is_finite()) {
            return Err("by gate must contain only finite values".to_string());
        }
    }
    Ok((row_offsets.len() - 1, penalty.nrows(), y.ncols()))
}

fn position_fit_design_for_slice(
    t: ArrayView1<'_, f64>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
    expected_cols: usize,
    batch_index: usize,
) -> Result<Array2<f64>, String> {
    let x = position_basis_design(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    if x.ncols() != expected_cols {
        return Err(format!(
            "position basis width mismatch in batch {batch_index}: penalty expects {expected_cols}, basis produced {}",
            x.ncols()
        ));
    }
    if let Some(by_values) = by {
        apply_by_gate(x.view(), by_values, by_start_col)
    } else {
        Ok(x)
    }
}

fn gaussian_reml_fit_positions_batched_streaming_impl(
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
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let (batch, p, d) = validate_position_batched_reml_common(
        t,
        y,
        row_offsets,
        knots_or_centers,
        penalty,
        weights,
        init_lambda,
        by,
        by_start_col,
    )?;

    // Build each ragged segment's basis independently. This makes it
    // impossible for the position-batched API to materialize the concatenated
    // n_total x p design, which is exactly the shape that becomes
    // operator-backed for large Duchon batches.
    let xtwx_phase: Vec<Result<Option<Array2<f64>>, String>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return Ok(None);
            }
            let x = position_fit_design_for_slice(
                t.slice(s![start..end]),
                knots_or_centers,
                basis_kind,
                basis_order,
                periodic,
                period,
                by.as_ref().map(|values| values.slice(s![start..end])),
                by_start_col,
                p,
                b,
            )?;
            let owned_weight: Array1<f64> = match weights.as_ref() {
                Some(w) => w.slice(s![start..end]).to_owned(),
                None => Array1::ones(end - start),
            };
            Ok(Some(gam::faer_ndarray::fast_xt_diag_x(
                &x.view(),
                &owned_weight,
            )))
        })
        .collect();

    let mut live_indices: Vec<usize> = Vec::with_capacity(batch);
    let mut live_xtwx: Vec<Array2<f64>> = Vec::with_capacity(batch);
    for (b, slot) in xtwx_phase.into_iter().enumerate() {
        if let Some(xtwx) = slot? {
            live_indices.push(b);
            live_xtwx.push(xtwx);
        }
    }
    let batched_caches = build_gaussian_reml_eigen_cache_batched(live_xtwx, penalty, None);
    let mut prebuilt_caches: Vec<Option<gam::gaussian_reml::GaussianRemlEigenCache>> =
        (0..batch).map(|_| None).collect();
    for (i, cache_result) in batched_caches.into_iter().enumerate() {
        if let Ok(cache) = cache_result {
            prebuilt_caches[live_indices[i]] = Some(cache);
        }
    }

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
            let x = position_fit_design_for_slice(
                t.slice(s![start..end]),
                knots_or_centers,
                basis_kind,
                basis_order,
                periodic,
                period,
                by.as_ref().map(|values| values.slice(s![start..end])),
                by_start_col,
                p,
                b,
            )?;
            let weight_slice = weights.as_ref().map(|w| w.slice(s![start..end]));
            let cache_ref = prebuilt_caches[b].as_ref();
            match gaussian_reml_multi_closed_form_with_cache(
                x.view(),
                y.slice(s![start..end, ..]),
                penalty,
                weight_slice,
                init_lambda,
                cache_ref,
            ) {
                Ok(result) => Ok((b, Some(result))),
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!(
                    "batched position Gaussian REML fit {b} failed: {err}"
                )),
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
    let mut fitted = Array2::<f64>::zeros((t.len(), d));
    let mut sigma2 = Array2::<f64>::from_elem((batch, d), f64::NAN);
    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut cache_penalty_eigenvalues = Array2::<f64>::zeros((batch, p));
    let mut cache_eigenvectors = Array3::<f64>::zeros((batch, p, p));
    let mut cache_coefficient_basis = Array3::<f64>::zeros((batch, p, p));
    let mut cache_xtwx_fingerprints = Array1::<u64>::zeros(batch);
    let mut cache_penalty_fingerprints = Array1::<u64>::zeros(batch);
    let mut cache_logdet_xtwx = Array1::<f64>::zeros(batch);
    let mut cache_logdet_penalty_positive = Array1::<f64>::zeros(batch);
    let mut cache_penalty_ranks = Array1::<i64>::zeros(batch);
    let mut cache_nullities = Array1::<i64>::zeros(batch);

    for result in fit_results {
        let (b, fit) = result?;
        if let Some(fit) = fit {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = if fit.lambda.is_finite() && fit.reml_score.is_finite() {
                "ok".to_string()
            } else {
                "diverged".to_string()
            };
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
            cache_penalty_eigenvalues
                .slice_mut(s![b, ..])
                .assign(&fit.cache.penalty_eigenvalues);
            cache_eigenvectors
                .slice_mut(s![b, .., ..])
                .assign(&fit.cache.eigenvectors);
            cache_coefficient_basis
                .slice_mut(s![b, .., ..])
                .assign(&fit.cache.coefficient_basis);
            cache_xtwx_fingerprints[b] = fit.cache.xtwx_fingerprint;
            cache_penalty_fingerprints[b] = fit.cache.penalty_fingerprint;
            cache_logdet_xtwx[b] = fit.cache.logdet_xtwx;
            cache_logdet_penalty_positive[b] = fit.cache.logdet_penalty_positive;
            cache_penalty_ranks[b] = fit.cache.penalty_rank as i64;
            cache_nullities[b] = fit.cache.nullity as i64;
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
        cache_penalty_eigenvalues,
        cache_eigenvectors,
        cache_coefficient_basis,
        cache_xtwx_fingerprints,
        cache_penalty_fingerprints,
        cache_logdet_xtwx,
        cache_logdet_penalty_positive,
        cache_penalty_ranks,
        cache_nullities,
    })
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
    grad_edf: Option<ArrayView1<'_, f64>>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
    forward_fits: Option<&[Option<gam::gaussian_reml::GaussianRemlMultiResult>]>,
) -> Result<BatchedPositionGaussianRemlBackwardResult, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let (batch, p, d) = validate_position_batched_reml_common(
        t,
        y,
        row_offsets,
        knots_or_centers,
        penalty,
        weights,
        init_lambda,
        by,
        by_start_col,
    )?;
    validate_batched_reml_upstreams(
        batch,
        p,
        d,
        t.len(),
        grad_lambda,
        grad_coefficients,
        grad_fitted,
        grad_reml_score,
        grad_edf,
    )?;

    if let Some(fits) = forward_fits {
        if fits.len() != batch {
            return Err(format!(
                "forward_state fit count mismatch: expected {batch}, got {}",
                fits.len()
            ));
        }
    }
    type PositionBackwardParts = (
        Array1<f64>,
        Array2<f64>,
        Array2<f64>,
        Array1<f64>,
        Option<Array1<f64>>,
    );
    let results: Vec<Result<(usize, Option<PositionBackwardParts>), String>> = (0..batch)
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
            let upstream_edf = grad_edf.as_ref().map_or(0.0, |g| g[b]);
            let upstream_coefficients = grad_coefficients.as_ref().map(|g| g.slice(s![b, .., ..]));
            let upstream_fitted = grad_fitted.as_ref().map(|g| g.slice(s![start..end, ..]));
            let x_base = position_basis_design(
                t.slice(s![start..end]),
                knots_or_centers,
                basis_kind,
                basis_order,
                periodic,
                period,
            )?;
            if x_base.ncols() != p {
                return Err(format!(
                    "position basis width mismatch in batch {b}: penalty expects {p}, basis produced {}",
                    x_base.ncols()
                ));
            }
            let basis_derivative = position_basis_derivative(
                t.slice(s![start..end]),
                knots_or_centers,
                basis_kind,
                basis_order,
                periodic,
                period,
            )?;
            if basis_derivative.dim() != x_base.dim() {
                return Err(format!(
                    "basis derivative shape mismatch in batch {b}: basis is {}x{} but dX/dt is {}x{}",
                    x_base.nrows(),
                    x_base.ncols(),
                    basis_derivative.nrows(),
                    basis_derivative.ncols()
                ));
            }
            let by_slice = by.as_ref().map(|values| values.slice(s![start..end]));
            let gated_x;
            let x_slice = if let Some(by_values) = by_slice {
                gated_x = apply_by_gate(x_base.view(), by_values, by_start_col)?;
                gated_x.view()
            } else {
                x_base.view()
            };
            let y_slice = y.slice(s![start..end, ..]);
            let backward_result = if let Some(fits) = forward_fits {
                match fits[b].as_ref() {
                    Some(fit) => gaussian_reml_multi_closed_form_backward_from_fit(
                        x_slice,
                        y_slice,
                        penalty,
                        weight_slice,
                        fit,
                        upstream_lambda,
                        upstream_coefficients,
                        upstream_fitted,
                        upstream_reml_score,
                        upstream_edf,
                    ),
                    None => return Ok((b, None)),
                }
            } else {
                gaussian_reml_multi_closed_form_backward(
                    x_slice,
                    y_slice,
                    penalty,
                    weight_slice,
                    init_lambda,
                    upstream_lambda,
                    upstream_coefficients,
                    upstream_fitted,
                    upstream_reml_score,
                    upstream_edf,
                )
            };
            match backward_result {
                Ok(backward) => {
                    let (grad_x, grad_by) = if let Some(by_values) = by_slice {
                        let (grad_x, grad_by) = apply_by_gate_backward(
                            x_base.view(),
                            by_values,
                            by_start_col,
                            backward.grad_x.view(),
                        )?;
                        (grad_x, Some(grad_by))
                    } else {
                        (backward.grad_x, None)
                    };
                    let grad_t = contract_position_gradient(grad_x.view(), basis_derivative.view())?;
                    Ok((
                        b,
                        Some((
                            grad_t,
                            backward.grad_y,
                            backward.grad_penalty,
                            backward.grad_weights,
                            grad_by,
                        )),
                    ))
                }
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!(
                    "batched position Gaussian REML backward {b} failed: {err}"
                )),
            }
        })
        .collect();

    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut grad_t = Array1::<f64>::zeros(t.len());
    let mut grad_y = Array2::<f64>::zeros(y.dim());
    let mut grad_penalty = Array2::<f64>::zeros(penalty.dim());
    let mut grad_weights = Array1::<f64>::zeros(t.len());
    let mut grad_by = by.map(|_| Array1::<f64>::zeros(t.len()));
    for result in results {
        let (b, backward) = result?;
        if let Some((
            batch_grad_t,
            batch_grad_y,
            batch_grad_penalty,
            batch_grad_weights,
            batch_grad_by,
        )) = backward
        {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = "ok".to_string();
            grad_t.slice_mut(s![start..end]).assign(&batch_grad_t);
            grad_y.slice_mut(s![start..end, ..]).assign(&batch_grad_y);
            grad_penalty += &batch_grad_penalty;
            grad_weights
                .slice_mut(s![start..end])
                .assign(&batch_grad_weights);
            if let (Some(target), Some(source)) = (grad_by.as_mut(), batch_grad_by.as_ref()) {
                target.slice_mut(s![start..end]).assign(source);
            }
        }
    }

    Ok(BatchedPositionGaussianRemlBackwardResult {
        statuses,
        grad_t,
        grad_y,
        grad_penalty,
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
        other => Err(format!(
            "normalized_position_basis_kind returned an unsupported basis name: {other}"
        )),
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
        other => Err(format!(
            "normalized_position_basis_kind returned an unsupported basis name: {other}"
        )),
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
    periodic_bspline_basis_dense_via_spec(t, (left, right), degree, num_basis)
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
    periodic_position_domain(knots, period)?;
    validate_vector("t", t)?;
    Err(
        "periodic B-spline first-derivative as a dense (N, K) matrix is no longer exposed; \
         use periodic_bspline_input_location_first_derivative for the (N, K, 1) jet"
            .to_string(),
    )
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
        if let Some(period) = period {
            if !period.is_finite() || period <= 0.0 {
                return Err(format!(
                    "{label} period must be finite and positive; got {period}"
                ));
            }
            if (implied - period).abs() > 1.0e-10 * period.max(1.0) {
                return Err(format!(
                    "{label} periodic support range ({implied}) must match explicit period ({period})"
                ));
            }
        } else if label != "duchon" {
            return Err(format!(
                "{label} periodic position basis requires an explicit period"
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
    detach_py_result(py, "posterior_predict_table", move || {
        posterior_predict_table_impl(&model_bytes, headers, rows, samples_flat, n_draws, n_coeffs)
    })
}

#[pyfunction]
fn posterior_predict_bands_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
    level: f64,
) -> PyResult<String> {
    detach_py_result(py, "posterior_predict_bands_table", move || {
        posterior_predict_bands_table_impl(
            &model_bytes,
            headers,
            rows,
            samples_flat,
            n_draws,
            n_coeffs,
            level,
        )
    })
}

#[pyfunction]
fn posterior_eta_bands(
    py: Python<'_>,
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    family_kind: String,
    level: f64,
) -> PyResult<String> {
    detach_py_result(py, "posterior_eta_bands", move || {
        posterior_eta_bands_impl(eta_flat, n_draws, n_rows, &family_kind, level)
    })
}

#[pyfunction]
fn posterior_credible_interval(
    py: Python<'_>,
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
    level: f64,
) -> PyResult<Vec<f64>> {
    detach_py_result(py, "posterior_credible_interval", move || {
        posterior_credible_interval_impl(samples_flat, n_draws, n_coeffs, level)
    })
}

#[pyfunction]
fn apply_inverse_link_array(
    py: Python<'_>,
    eta: Vec<f64>,
    family_kind: String,
) -> PyResult<Vec<f64>> {
    detach_py_result(py, "apply_inverse_link_array", move || {
        apply_inverse_link_vec(&eta, &family_kind)
    })
}

#[pyfunction]
fn summary_json(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<String> {
    detach_py_result(py, "summary_json", move || summary_json_impl(&model_bytes))
}

#[pyfunction]
fn summary_payload_from_model(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<PyObject> {
    let payload = detach_py_result(py, "summary_payload_from_model", move || {
        summary_payload_value_from_model_bytes(&model_bytes)
    })?;
    json_object_to_py_dict(py, payload)
}

fn summary_payload_value_from_model_bytes(
    model_bytes: &[u8],
) -> Result<serde_json::Value, String> {
    let summary_json = summary_json_impl(model_bytes)?;
    serde_json::from_str(&summary_json).map_err(|err| format!("invalid model summary JSON: {err}"))
}

fn json_object_to_py_dict(py: Python<'_>, value: serde_json::Value) -> PyResult<PyObject> {
    let serde_json::Value::Object(items) = value else {
        return Err(py_value_error(
            "model summary payload must be a JSON object".to_string(),
        ));
    };
    let out = PyDict::new(py);
    for (key, value) in items {
        let py_value = json_value_to_py(py, value)?;
        out.set_item(key, py_value.bind(py))?;
    }
    Ok(out.unbind().into_any())
}

fn json_value_to_py(py: Python<'_>, value: serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(value) => value.into_py_any(py),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                return value.into_py_any(py);
            }
            if let Some(value) = value.as_u64() {
                return value.into_py_any(py);
            }
            let value = value
                .as_f64()
                .ok_or_else(|| py_value_error("JSON number is not representable".to_string()))?;
            value.into_py_any(py)
        }
        serde_json::Value::String(value) => value.into_py_any(py),
        serde_json::Value::Array(values) => {
            let out = PyList::empty(py);
            for value in values {
                let py_value = json_value_to_py(py, value)?;
                out.append(py_value.bind(py))?;
            }
            Ok(out.unbind().into_any())
        }
        serde_json::Value::Object(values) => {
            let out = PyDict::new(py);
            for (key, value) in values {
                let py_value = json_value_to_py(py, value)?;
                out.set_item(key, py_value.bind(py))?;
            }
            Ok(out.unbind().into_any())
        }
    }
}

#[pyfunction]
fn summary_repr(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    let mut fields = Vec::new();
    for key in ["formula", "family_name", "model_class", "deviance", "reml_score"] {
        if let Some(value) = payload.get_item(key)? {
            let repr = value.repr()?.extract::<String>()?;
            fields.push(format!("{key}={repr}"));
        }
    }
    Ok(format!("Summary({})", fields.join(", ")))
}

#[pyfunction]
fn summary_html(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    // Pure presentation layer; no math.
    let mut rows = String::new();
    for (key, value) in payload.iter() {
        let key_text = key.str()?.extract::<String>()?;
        if key_text == "coefficients" || key_text == "covariance_flat" {
            continue;
        }
        rows.push_str("<tr>");
        rows.push_str(
            "<th style='text-align:left;padding:0.25rem 0.75rem 0.25rem 0;'>",
        );
        rows.push_str(&summary_html_escape(&key_text));
        rows.push_str("</th><td style='padding:0.25rem 0;'>");
        rows.push_str(&summary_html_escape(&summary_render_value(&value)?));
        rows.push_str("</td></tr>");
    }
    let coefficient_table = summary_render_coefficients_html(payload)?;
    Ok(format!(
        "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>\
         <h3 style='margin:0 0 0.5rem 0;'>Model Summary</h3>\
         <table style='border-collapse:collapse;'>{rows}</table>\
         {coefficient_table}\
         </div>"
    ))
}

const SUMMARY_HTML_COEFFICIENT_LIMIT: usize = 50;
const SUMMARY_VALUE_PREVIEW_LIMIT: usize = 6;

fn summary_render_value(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(float_value) = value.cast::<PyFloat>() {
        return Ok(summary_format_float(float_value.extract::<f64>()?));
    }
    if let Ok(mapping) = value.cast::<PyDict>() {
        return summary_render_mapping_value(mapping);
    }
    if let Ok(sequence) = value.cast::<PyList>() {
        return summary_render_list_value(sequence);
    }
    if let Ok(sequence) = value.cast::<PyTuple>() {
        return summary_render_tuple_value(sequence);
    }
    value.str()?.extract::<String>()
}

fn summary_render_mapping_value(value: &Bound<'_, PyDict>) -> PyResult<String> {
    if value.is_empty() {
        return Ok("{}".to_string());
    }
    let mut parts = Vec::new();
    for (index, (key, item_value)) in value.iter().enumerate() {
        if index >= SUMMARY_VALUE_PREVIEW_LIMIT {
            break;
        }
        parts.push(format!(
            "{}: {}",
            key.str()?.extract::<String>()?,
            summary_render_value(&item_value)?
        ));
    }
    let suffix = if value.len() <= SUMMARY_VALUE_PREVIEW_LIMIT {
        String::new()
    } else {
        format!(", ... ({} total)", value.len())
    };
    Ok(format!("{{{}{}}}", parts.join(", "), suffix))
}

fn summary_render_list_value(value: &Bound<'_, PyList>) -> PyResult<String> {
    if value.is_empty() {
        return Ok("[]".to_string());
    }
    let preview_len = value.len().min(SUMMARY_VALUE_PREVIEW_LIMIT);
    let mut parts = Vec::with_capacity(preview_len);
    for index in 0..preview_len {
        parts.push(summary_render_value(&value.get_item(index)?)?);
    }
    let suffix = if value.len() <= SUMMARY_VALUE_PREVIEW_LIMIT {
        String::new()
    } else {
        format!(", ... ({} total)", value.len())
    };
    Ok(format!("[{}{}]", parts.join(", "), suffix))
}

fn summary_render_tuple_value(value: &Bound<'_, PyTuple>) -> PyResult<String> {
    if value.is_empty() {
        return Ok("[]".to_string());
    }
    let preview_len = value.len().min(SUMMARY_VALUE_PREVIEW_LIMIT);
    let mut parts = Vec::with_capacity(preview_len);
    for index in 0..preview_len {
        parts.push(summary_render_value(&value.get_item(index)?)?);
    }
    let suffix = if value.len() <= SUMMARY_VALUE_PREVIEW_LIMIT {
        String::new()
    } else {
        format!(", ... ({} total)", value.len())
    };
    Ok(format!("[{}{}]", parts.join(", "), suffix))
}

fn summary_render_coefficients_html(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    let Some(coefficients_any) = payload.get_item("coefficients")? else {
        return Ok(String::new());
    };
    let Ok(coefficients) = coefficients_any.cast::<PyList>() else {
        return Ok(String::new());
    };
    if coefficients.is_empty() {
        return Ok(String::new());
    }

    let columns = summary_coefficient_columns(coefficients)?;
    let mut header_cells = String::new();
    for column in &columns {
        header_cells.push_str(
            "<th style='text-align:right;padding:0.25rem 0.75rem;\
             border-bottom:1px solid #ddd;'>",
        );
        header_cells.push_str(&summary_html_escape(column));
        header_cells.push_str("</th>");
    }

    let row_limit = coefficients.len().min(SUMMARY_HTML_COEFFICIENT_LIMIT);
    let mut body_rows = String::new();
    for index in 0..row_limit {
        let row_any = coefficients.get_item(index)?;
        let row = row_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("summary coefficient rows must be dictionaries"))?;
        body_rows.push_str("<tr>");
        for column in &columns {
            let rendered = match row.get_item(column.as_str())? {
                Some(value) => summary_render_value(&value)?,
                None => String::new(),
            };
            body_rows.push_str("<td style='text-align:right;padding:0.25rem 0.75rem;'>");
            body_rows.push_str(&summary_html_escape(&rendered));
            body_rows.push_str("</td>");
        }
        body_rows.push_str("</tr>");
    }

    let note = if coefficients.len() > SUMMARY_HTML_COEFFICIENT_LIMIT {
        format!(
            "<p style='margin:0.25rem 0 0 0;color:#666;'>Showing first {} of {} coefficients.</p>",
            SUMMARY_HTML_COEFFICIENT_LIMIT,
            coefficients.len()
        )
    } else {
        String::new()
    };

    Ok(format!(
        "<h4 style='margin:1rem 0 0.35rem 0;'>Coefficients</h4>\
         <table style='border-collapse:collapse;'>\
         <thead><tr>{header_cells}</tr></thead>\
         <tbody>{body_rows}</tbody>\
         </table>\
         {note}"
    ))
}

fn summary_coefficient_columns(coefficients: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
    let mut columns = Vec::new();
    for index in 0..coefficients.len() {
        let row_any = coefficients.get_item(index)?;
        let row = row_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("summary coefficient rows must be dictionaries"))?;
        for (key, _) in row.iter() {
            let column = key.str()?.extract::<String>()?;
            if !columns.iter().any(|existing| existing == &column) {
                columns.push(column);
            }
        }
    }
    Ok(columns)
}

fn summary_format_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value == f64::INFINITY {
        return "inf".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }

    let exponent = value.abs().log10().floor() as i32;
    let mut out = if !(-4..6).contains(&exponent) {
        let raw = format!("{:.5e}", value);
        summary_normalize_exponent(&raw)
    } else {
        let places = (6 - exponent - 1).max(0) as usize;
        summary_trim_float(format!("{:.*}", places, value))
    };
    if out == "-0" {
        out = "0".to_string();
    }
    out
}

fn summary_normalize_exponent(raw: &str) -> String {
    let Some((mantissa, exponent)) = raw.split_once('e') else {
        return raw.to_string();
    };
    let mantissa = summary_trim_float(mantissa.to_string());
    let (sign, digits) = if let Some(rest) = exponent.strip_prefix('-') {
        ('-', rest)
    } else if let Some(rest) = exponent.strip_prefix('+') {
        ('+', rest)
    } else {
        ('+', exponent)
    };
    let digits = digits.trim_start_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let padded = if digits.len() == 1 {
        format!("0{digits}")
    } else {
        digits.to_string()
    };
    format!("{mantissa}e{sign}{padded}")
}

fn summary_trim_float(mut value: String) -> String {
    if value.contains('.') {
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
    }
    value
}

fn summary_html_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

#[pyfunction]
fn coefficient_state_json(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<String> {
    detach_py_result(py, "coefficient_state_json", move || {
        coefficient_state_json_impl(&model_bytes)
    })
}

#[pyfunction]
fn term_blocks_for_model(
    model_bytes: Vec<u8>,
) -> PyResult<Vec<(String, String, usize, usize)>> {
    term_blocks_for_model_impl(&model_bytes).map_err(PyValueError::new_err)
}

#[pyfunction]
fn difference_smooth_json(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    request_json: String,
) -> PyResult<String> {
    detach_py_result(py, "difference_smooth_json", move || {
        difference_smooth_json_impl(&model_bytes, &request_json)
    })
}

#[pyfunction]
fn difference_smooth_rows(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    request_json: String,
) -> PyResult<PyObject> {
    let rows = detach_py_result(py, "difference_smooth_rows", move || {
        let raw = difference_smooth_json_impl(&model_bytes, &request_json)?;
        serde_json::from_str::<Vec<serde_json::Map<String, serde_json::Value>>>(&raw)
            .map_err(|err| format!("failed to parse difference_smooth rows json: {err}"))
    })?;
    difference_smooth_rows_to_py(py, rows)
}

fn difference_smooth_rows_to_py(
    py: Python<'_>,
    rows: Vec<serde_json::Map<String, serde_json::Value>>,
) -> PyResult<PyObject> {
    let out = PyList::empty(py);
    for row in rows {
        let py_row = PyDict::new(py);
        for (key, value) in row {
            py_dict_set_json_value(py, &py_row, &key, value)?;
        }
        out.append(py_row)?;
    }
    Ok(out.unbind().into_any())
}

fn py_dict_set_json_value(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
    value: serde_json::Value,
) -> PyResult<()> {
    match value {
        serde_json::Value::Null => dict.set_item(key, py.None()),
        serde_json::Value::Bool(value) => dict.set_item(key, value),
        serde_json::Value::Number(value) => py_dict_set_json_number(dict, key, value),
        serde_json::Value::String(value) => dict.set_item(key, value),
        serde_json::Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                py_list_append_json_value(py, &list, value)?;
            }
            dict.set_item(key, list)
        }
        serde_json::Value::Object(values) => {
            let nested = PyDict::new(py);
            for (nested_key, nested_value) in values {
                py_dict_set_json_value(py, &nested, &nested_key, nested_value)?;
            }
            dict.set_item(key, nested)
        }
    }
}

fn py_dict_set_json_number(
    dict: &Bound<'_, PyDict>,
    key: &str,
    value: serde_json::Number,
) -> PyResult<()> {
    if let Some(value) = value.as_i64() {
        dict.set_item(key, value)
    } else if let Some(value) = value.as_u64() {
        dict.set_item(key, value)
    } else if let Some(value) = value.as_f64() {
        dict.set_item(key, value)
    } else {
        Err(PyValueError::new_err(
            "difference_smooth row contains an unsupported JSON number",
        ))
    }
}

fn py_list_append_json_value(
    py: Python<'_>,
    list: &Bound<'_, PyList>,
    value: serde_json::Value,
) -> PyResult<()> {
    match value {
        serde_json::Value::Null => list.append(py.None()),
        serde_json::Value::Bool(value) => list.append(value),
        serde_json::Value::Number(value) => py_list_append_json_number(list, value),
        serde_json::Value::String(value) => list.append(value),
        serde_json::Value::Array(values) => {
            let nested = PyList::empty(py);
            for value in values {
                py_list_append_json_value(py, &nested, value)?;
            }
            list.append(nested)
        }
        serde_json::Value::Object(values) => {
            let nested = PyDict::new(py);
            for (nested_key, nested_value) in values {
                py_dict_set_json_value(py, &nested, &nested_key, nested_value)?;
            }
            list.append(nested)
        }
    }
}

fn py_list_append_json_number(
    list: &Bound<'_, PyList>,
    value: serde_json::Number,
) -> PyResult<()> {
    if let Some(value) = value.as_i64() {
        list.append(value)
    } else if let Some(value) = value.as_u64() {
        list.append(value)
    } else if let Some(value) = value.as_f64() {
        list.append(value)
    } else {
        Err(PyValueError::new_err(
            "difference_smooth row contains an unsupported JSON number",
        ))
    }
}

#[pyfunction]
fn cross_fit_shared_precision_groups_json(
    py: Python<'_>,
    request_json: String,
) -> PyResult<String> {
    detach_py_result(py, "cross_fit_shared_precision_groups_json", move || {
        cross_fit_shared_precision_groups_json_impl(&request_json)
    })
}

#[pyfunction]
fn check_json(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> PyResult<String> {
    detach_py_result(py, "check_json", move || {
        check_json_impl(&model_bytes, headers, rows)
    })
}

#[pyfunction]
fn check_payload_from_model(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> PyResult<PyObject> {
    let payload = detach_py_result(py, "check_payload_from_model", move || {
        let check_json = check_json_impl(&model_bytes, headers, rows)?;
        serde_json::from_str::<serde_json::Value>(&check_json)
            .map_err(|err| format!("invalid schema check JSON: {err}"))
    })?;
    json_object_to_py_dict(py, payload)
}

#[pyfunction]
fn report_html(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<String> {
    detach_py_result(py, "report_html", move || report_html_impl(&model_bytes))
}

#[pyfunction]
fn compute_residuals<'py>(
    py: Python<'py>,
    observed: PyReadonlyArray1<'py, f64>,
    predicted_mean: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let observed_values = observed.as_array();
    let predicted_values = predicted_mean.as_array();
    if observed_values.len() != predicted_values.len() {
        return Err(py_value_error(format!(
            "compute_residuals length mismatch: observed has {} values but predicted mean has {}",
            observed_values.len(),
            predicted_values.len()
        )));
    }

    let residuals = observed_values
        .iter()
        .zip(predicted_values.iter())
        .map(|(obs, pred)| *obs - *pred)
        .collect::<Vec<_>>();
    Ok(Array1::from_vec(residuals).into_pyarray(py).unbind())
}

#[pyfunction]
fn diagnostics_from_predictions(
    py: Python<'_>,
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let diagnostics =
        gam::inference::diagnostics::diagnostics_from_predictions(&observed, &predicted_mean)
            .map_err(py_value_error)?;

    let metrics = PyDict::new(py);
    metrics.set_item("n_obs", diagnostics.n_obs)?;
    metrics.set_item("mae", diagnostics.mae)?;
    metrics.set_item("rmse", diagnostics.rmse)?;
    metrics.set_item("bias", diagnostics.bias)?;
    if let Some(r_squared) = diagnostics.r_squared {
        metrics.set_item("r_squared", r_squared)?;
    }

    let out = PyDict::new(py);
    out.set_item("residuals", PyList::new(py, diagnostics.residuals)?)?;
    out.set_item("metrics", metrics)?;
    Ok(out.unbind())
}

fn benchmark_auc_score(observed: &[f64], predicted_mean: &[f64]) -> PyResult<f64> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "auc length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    let mut pairs: Vec<(f64, bool)> = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| (p, y > 0.5))
        .collect();
    let n_pos = pairs.iter().filter(|(_, is_pos)| *is_pos).count();
    let n_neg = pairs.len().saturating_sub(n_pos);
    if n_pos == 0 || n_neg == 0 {
        return Ok(0.5);
    }
    pairs.sort_by(|(a, _), (b, _)| a.total_cmp(b));

    let mut concordant = 0.0;
    let mut negatives_below = 0usize;
    let mut i = 0usize;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let pos_in_group = pairs[i..j]
            .iter()
            .filter(|(_, is_pos)| *is_pos)
            .count();
        let neg_in_group = (j - i).saturating_sub(pos_in_group);
        concordant += (pos_in_group * negatives_below) as f64;
        concordant += 0.5 * (pos_in_group * neg_in_group) as f64;
        negatives_below += neg_in_group;
        i = j;
    }
    Ok(concordant / ((n_pos * n_neg) as f64))
}

fn benchmark_binary_logloss(observed: &[f64], predicted_mean: &[f64]) -> PyResult<f64> {
    benchmark_binary_logloss_eps(observed, predicted_mean, 1.0e-12)
}

fn benchmark_binary_logloss_eps(
    observed: &[f64],
    predicted_mean: &[f64],
    eps: f64,
) -> PyResult<f64> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "logloss length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() {
        return Err(PyValueError::new_err("logloss requires at least one row"));
    }
    let loss_sum = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(eps, 1.0 - eps);
            -(y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln())
        })
        .sum::<f64>();
    Ok(loss_sum / observed.len() as f64)
}

fn benchmark_exp_saturated(x: f64) -> f64 {
    if x >= 709.0 {
        f64::INFINITY
    } else if x <= -745.0 {
        0.0
    } else {
        x.exp()
    }
}

fn benchmark_nagelkerke_r2(
    observed: &[f64],
    predicted_mean: &[f64],
    train_observed: &[f64],
) -> PyResult<Option<f64>> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "nagelkerke length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() || train_observed.is_empty() {
        return Ok(None);
    }
    let null_mean = train_observed.iter().sum::<f64>() / train_observed.len() as f64;
    if !null_mean.is_finite() || null_mean <= 0.0 || null_mean >= 1.0 {
        return Ok(None);
    }
    let eps = 1.0e-12;
    let log_null = null_mean.ln();
    let log_not_null = (1.0 - null_mean).ln();
    let ll_null = observed
        .iter()
        .map(|&y| y * log_null + (1.0 - y) * log_not_null)
        .sum::<f64>();
    let ll_model = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(eps, 1.0 - eps);
            y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln()
        })
        .sum::<f64>();
    let n = observed.len() as f64;
    let r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * (ll_null - ll_model));
    let max_r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * ll_null);
    if !r2_cs.is_finite() || !max_r2_cs.is_finite() || max_r2_cs <= 0.0 {
        return Ok(None);
    }
    Ok(Some(r2_cs / max_r2_cs))
}

fn benchmark_nagelkerke_r2_with_null_mean(
    observed: &[f64],
    predicted_mean: &[f64],
    null_mean: f64,
    eps: f64,
) -> PyResult<Option<f64>> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "nagelkerke length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() {
        return Ok(None);
    }
    if !null_mean.is_finite() || null_mean <= 0.0 || null_mean >= 1.0 {
        return Ok(None);
    }
    let log_null = null_mean.ln();
    let log_not_null = (1.0 - null_mean).ln();
    let ll_null = observed
        .iter()
        .map(|&y| y * log_null + (1.0 - y) * log_not_null)
        .sum::<f64>();
    let ll_model = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(eps, 1.0 - eps);
            y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln()
        })
        .sum::<f64>();
    let n = observed.len() as f64;
    let r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * (ll_null - ll_model));
    let max_r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * ll_null);
    if !r2_cs.is_finite() || !max_r2_cs.is_finite() || max_r2_cs <= 0.0 {
        return Ok(None);
    }
    Ok(Some(r2_cs / max_r2_cs))
}

fn benchmark_gaussian_logloss(
    observed: &[f64],
    predicted_mean: &[f64],
    sigma: &[f64],
) -> PyResult<f64> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "gaussian logloss length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() {
        return Err(PyValueError::new_err(
            "gaussian logloss requires at least one row",
        ));
    }
    if sigma.len() != 1 && sigma.len() != observed.len() {
        return Err(PyValueError::new_err(format!(
            "sigma length must be 1 or {}, got {}",
            observed.len(),
            sigma.len()
        )));
    }
    let eps = 1.0e-12;
    let two_pi = 2.0 * std::f64::consts::PI;
    let loss_sum = observed
        .iter()
        .zip(predicted_mean.iter())
        .enumerate()
        .map(|(idx, (&y, &mu))| {
            let raw_sigma = if sigma.len() == 1 { sigma[0] } else { sigma[idx] };
            let sigma_use = raw_sigma.max(eps);
            let var = sigma_use * sigma_use;
            0.5 * (two_pi * var).ln() + ((y - mu) * (y - mu)) / (2.0 * var)
        })
        .sum::<f64>();
    Ok(loss_sum / observed.len() as f64)
}

#[pyfunction]
fn benchmark_prediction_metrics(
    py: Python<'_>,
    family: String,
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
    train_observed: Vec<f64>,
    sigma: Option<Vec<f64>>,
) -> PyResult<Py<PyDict>> {
    let diagnostics =
        gam::inference::diagnostics::diagnostics_from_predictions(&observed, &predicted_mean)
            .map_err(py_value_error)?;
    let metrics = PyDict::new(py);
    match family.as_str() {
        "binomial" => {
            metrics.set_item("auc", benchmark_auc_score(&observed, &predicted_mean)?)?;
            metrics.set_item("brier", diagnostics.rmse * diagnostics.rmse)?;
            metrics.set_item(
                "logloss",
                benchmark_binary_logloss(&observed, &predicted_mean)?,
            )?;
            if let Some(nagelkerke_r2) =
                benchmark_nagelkerke_r2(&observed, &predicted_mean, &train_observed)?
            {
                metrics.set_item("nagelkerke_r2", nagelkerke_r2)?;
            }
        }
        "gaussian" => {
            metrics.set_item("r2", diagnostics.r_squared.unwrap_or(0.0))?;
            metrics.set_item("rmse", diagnostics.rmse)?;
            metrics.set_item("mae", diagnostics.mae)?;
            metrics.set_item("mse", diagnostics.rmse * diagnostics.rmse)?;
            if let Some(sigma_values) = sigma {
                metrics.set_item(
                    "logloss",
                    benchmark_gaussian_logloss(&observed, &predicted_mean, &sigma_values)?,
                )?;
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported benchmark metric family: {other}"
            )));
        }
    }
    Ok(metrics.unbind())
}

fn benchmark_pr_auc_score(observed: &[f64], predicted_mean: &[f64]) -> PyResult<f64> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "pr_auc length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    let mut pairs: Vec<(f64, bool)> = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| (p, y > 0.5))
        .collect();
    let n_pos = pairs.iter().filter(|(_, is_pos)| *is_pos).count();
    if n_pos == 0 {
        return Ok(0.0);
    }
    pairs.sort_by(|(a, _), (b, _)| b.total_cmp(a));
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut prev_precision = 1.0;
    let mut prev_recall = 0.0;
    let mut area = 0.0;
    for (_score, is_pos) in pairs {
        if is_pos {
            tp += 1;
        } else {
            fp += 1;
        }
        let precision = tp as f64 / (tp + fp).max(1) as f64;
        let recall = tp as f64 / n_pos as f64;
        area += 0.5 * (precision + prev_precision) * (recall - prev_recall);
        prev_precision = precision;
        prev_recall = recall;
    }
    Ok(area)
}

fn benchmark_ece_score(observed: &[f64], predicted_mean: &[f64], n_bins: usize) -> PyResult<f64> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "ece length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() {
        return Err(PyValueError::new_err("ece requires at least one row"));
    }
    let mut bins = vec![(0usize, 0.0, 0.0); n_bins.max(1)];
    let bins_len = bins.len();
    for (&y, &p) in observed.iter().zip(predicted_mean.iter()) {
        if !y.is_finite() || !p.is_finite() {
            return Err(PyValueError::new_err("ece inputs must be finite"));
        }
        let mut idx = (p.clamp(0.0, 1.0) * bins_len as f64).floor() as usize;
        if idx >= bins_len {
            idx = bins_len - 1;
        }
        bins[idx].0 += 1;
        bins[idx].1 += y;
        bins[idx].2 += p;
    }
    let n = observed.len() as f64;
    Ok(bins
        .into_iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|(count, y_sum, p_sum)| {
            let c = count as f64;
            (c / n) * ((y_sum / c) - (p_sum / c)).abs()
        })
        .sum())
}

#[pyfunction]
fn binary_auc(observed: Vec<f64>, predicted_mean: Vec<f64>) -> PyResult<f64> {
    benchmark_auc_score(&observed, &predicted_mean)
}

#[pyfunction]
fn binary_brier(observed: Vec<f64>, predicted_mean: Vec<f64>) -> PyResult<f64> {
    let diagnostics =
        gam::inference::diagnostics::diagnostics_from_predictions(&observed, &predicted_mean)
            .map_err(py_value_error)?;
    Ok(diagnostics.rmse * diagnostics.rmse)
}

#[pyfunction]
fn classification_metrics(
    py: Python<'_>,
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
    train_prev: f64,
) -> PyResult<Py<PyDict>> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "classification_metrics length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    let diagnostics =
        gam::inference::diagnostics::diagnostics_from_predictions(&observed, &predicted_mean)
            .map_err(py_value_error)?;
    let metrics = PyDict::new(py);
    metrics.set_item("auc", benchmark_auc_score(&observed, &predicted_mean)?)?;
    metrics.set_item("pr_auc", benchmark_pr_auc_score(&observed, &predicted_mean)?)?;
    metrics.set_item("brier", diagnostics.rmse * diagnostics.rmse)?;
    metrics.set_item(
        "logloss",
        benchmark_binary_logloss(&observed, &predicted_mean)?,
    )?;
    if train_prev > 0.0 && train_prev < 1.0 {
        let train_observed = vec![train_prev];
        let null_mean = train_observed[0];
        let eps = 1.0e-12;
        let ll_null = observed
            .iter()
            .map(|&y| y * null_mean.ln() + (1.0 - y) * (1.0 - null_mean).ln())
            .sum::<f64>();
        let ll_model = observed
            .iter()
            .zip(predicted_mean.iter())
            .map(|(&y, &p)| {
                let clipped = p.clamp(eps, 1.0 - eps);
                y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln()
            })
            .sum::<f64>();
        let n = observed.len() as f64;
        let r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * (ll_null - ll_model));
        let max_r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * ll_null);
        if r2_cs.is_finite() && max_r2_cs.is_finite() && max_r2_cs > 0.0 {
            metrics.set_item("nagelkerke_r2", r2_cs / max_r2_cs)?;
        } else {
            metrics.set_item("nagelkerke_r2", py.None())?;
        }
    } else {
        metrics.set_item("nagelkerke_r2", py.None())?;
    }
    metrics.set_item("ece", benchmark_ece_score(&observed, &predicted_mean, 20)?)?;
    Ok(metrics.unbind())
}

// =========================================================================
// LatentCoord input-location derivative helpers (thin pyffi wrappers).
//
// Each function here is a thin shim over a `*_first_derivative_nd` helper
// in `gam::terms::basis`. The helpers are analytic, closed-form chain rules
// of the underlying basis with respect to the *first* kernel argument
// (the latent coordinate `t`). They share the same analytic machinery the
// kernel-parameter chain (`SpatialLogKappaCoords`) uses, re-pointed at
// `t` instead of at the anisotropy / log-kappa.
// =========================================================================

/// `φ'(r_{n,k})` for the Duchon kernel at every `(row, center)` pair.
/// Returns `(n_rows, n_centers)`. Multiply by `(t − c)/r` at the call site
/// to get the per-row gradient — or use
/// [`crate::terms::latent_coord::LatentCoordValues::design_gradient_wrt_t`].
#[pyfunction(signature = (t, centers, length_scale = None, m = 2))]
fn duchon_input_location_first_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    length_scale: Option<f64>,
    m: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    if m == 0 {
        return Err(py_value_error(
            "duchon_input_location_first_derivative: m must be >= 1".to_string(),
        ));
    }
    let nullspace_order = duchon_nullspace_from_m(m);
    let out = duchon_radial_first_derivative_nd(
        t.as_array(),
        centers.as_array(),
        length_scale,
        nullspace_order,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    Ok(out.into_pyarray(py).unbind())
}

/// `(N, n_poly, d)` jet of the Duchon polynomial-nullspace tail (closed-form
/// monomial derivatives). Column ordering matches the
/// `monomial_basis_block` enumeration used by the Duchon design builder.
#[pyfunction(signature = (t, m = 2))]
fn duchon_polynomial_input_location_first_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    m: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    if m == 0 {
        return Err(py_value_error(
            "duchon_polynomial_input_location_first_derivative: m must be >= 1".to_string(),
        ));
    }
    let nullspace_order = duchon_nullspace_from_m(m);
    let jet = duchon_polynomial_first_derivative_nd(t.as_array(), nullspace_order);
    Ok(jet.into_pyarray(py).unbind())
}

/// `φ'(r_{n,k})` for the Matérn kernel at every `(row, center)` pair.
/// Returns `(n_rows, n_centers)`. `nu` accepted: `"1/2"`, `"3/2"`, `"5/2"`,
/// `"7/2"`, `"9/2"` (any whitespace stripped).
fn parse_matern_nu_py(context: &str, nu: &str) -> PyResult<MaternNu> {
    match nu.replace(char::is_whitespace, "").as_str() {
        "1/2" | "0.5" => Ok(MaternNu::Half),
        "3/2" | "1.5" => Ok(MaternNu::ThreeHalves),
        "5/2" | "2.5" => Ok(MaternNu::FiveHalves),
        "7/2" | "3.5" => Ok(MaternNu::SevenHalves),
        "9/2" | "4.5" => Ok(MaternNu::NineHalves),
        other => Err(py_value_error(format!(
            "{context}: nu must be one of '1/2','3/2','5/2','7/2','9/2'; got {other:?}"
        ))),
    }
}

#[pyfunction(signature = (t, centers, length_scale, nu = "3/2"))]
fn matern_input_location_first_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    length_scale: f64,
    nu: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let nu_parsed = parse_matern_nu_py("matern_input_location_first_derivative", nu)?;
    let out = matern_radial_first_derivative_nd(
        t.as_array(),
        centers.as_array(),
        length_scale,
        nu_parsed,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    Ok(out.into_pyarray(py).unbind())
}

/// Streaming closed-form `dK/dtheta` for Matérn basis values.
///
/// `target="log_kappa"` returns the global scale derivative. `target="aniso"`
/// requires `axis` and returns the derivative for that axis' log-scale.
#[pyfunction(signature = (
    data,
    centers,
    length_scale,
    nu = "3/2",
    target = "log_kappa",
    axis = None,
    aniso_log_scales = None,
    chunk_size = None
))]
fn matern_basis_gradient_streaming<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    length_scale: f64,
    nu: &str,
    target: &str,
    axis: Option<usize>,
    aniso_log_scales: Option<PyReadonlyArray1<'py, f64>>,
    chunk_size: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let nu_parsed = parse_matern_nu_py("matern_basis_gradient_streaming", nu)?;
    let target = match target
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "log_kappa" | "kappa" | "scale" => MaternBasisGradientTarget::LogKappa,
        "aniso" | "aniso_log_scale" | "axis" => {
            MaternBasisGradientTarget::AnisoLogScale(axis.ok_or_else(|| {
                py_value_error(
                    "matern_basis_gradient_streaming: axis is required for target='aniso'"
                        .to_string(),
                )
            })?)
        }
        other => {
            return Err(py_value_error(format!(
                "matern_basis_gradient_streaming: target must be 'log_kappa' or 'aniso'; got {other:?}"
            )));
        }
    };
    let aniso = aniso_log_scales
        .as_ref()
        .map(|values| values.as_slice())
        .transpose()
        .map_err(|err| py_value_error(format!("aniso_log_scales must be contiguous: {err}")))?;
    let evaluator = StreamingMaternBasisGradientEvaluator::new(
        centers.as_array(),
        length_scale,
        nu_parsed,
        aniso,
        chunk_size,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    let out = evaluator
        .evaluate(data.as_array(), target)
        .map_err(|err| py_value_error(err.to_string()))?;
    Ok(out.into_pyarray(py).unbind())
}

/// Closed-form `(N, n_centers, dim)` jet of the Sobolev sphere kernel
/// w.r.t. ambient coordinates `t_n ∈ S^{dim-1}`.
///
/// The returned jet is tangent-projected by default for the intrinsic
/// Riemannian gradient (`g − (g · t) t`) used by embedded sphere updates.
/// Pass `project_to_tangent = false` to receive the ambient gradient.
#[pyfunction(signature = (points, centers, penalty_order = 2, project_to_tangent = true))]
fn sphere_input_location_first_derivative<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    penalty_order: usize,
    project_to_tangent: bool,
) -> PyResult<Py<PyArray3<f64>>> {
    let jet = sphere_first_derivative_nd(
        points.as_array(),
        centers.as_array(),
        penalty_order,
        project_to_tangent,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    Ok(jet.into_pyarray(py).unbind())
}

/// Closed-form `(N, num_basis, 1)` jet of the cyclic-B-spline basis w.r.t.
/// the scalar latent coordinate. `t` is `(N, 1)`.
#[pyfunction]
fn periodic_bspline_input_location_first_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    data_range_left: f64,
    data_range_right: f64,
    degree: usize,
    num_basis: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    let jet = periodic_bspline_first_derivative_nd(
        t.as_array(),
        (data_range_left, data_range_right),
        degree,
        num_basis,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    Ok(jet.into_pyarray(py).unbind())
}

/// Closed-form `(N, K_total, n_axes)` tensor-product B-spline derivative
/// jet, via the product rule. `knots_concat` is the per-axis knot vectors
/// flattened together; `knot_offsets` is `[0, len_axis_0, len_axis_0 +
/// len_axis_1, ...]` (length `n_axes + 1`) — i.e. the standard CSR-style
/// concat. `degrees` is per-axis.
#[pyfunction]
fn bspline_tensor_input_location_first_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    knots_concat: PyReadonlyArray1<'py, f64>,
    knot_offsets: Vec<usize>,
    degrees: Vec<usize>,
) -> PyResult<Py<PyArray3<f64>>> {
    let n_axes = degrees.len();
    if knot_offsets.len() != n_axes + 1 {
        return Err(py_value_error(format!(
            "bspline_tensor_input_location_first_derivative: knot_offsets must have \
             length n_axes + 1 = {}, got {}",
            n_axes + 1,
            knot_offsets.len()
        )));
    }
    if t.as_array().ncols() != n_axes {
        return Err(py_value_error(format!(
            "bspline_tensor_input_location_first_derivative: t has {} cols but \
             degrees has {} entries",
            t.as_array().ncols(),
            n_axes
        )));
    }
    let knots_concat_view = knots_concat.as_array();
    // Slice the concatenated knot vector into per-axis views.
    let mut per_axis_views: Vec<ArrayView1<'_, f64>> = Vec::with_capacity(n_axes);
    for a in 0..n_axes {
        let lo = knot_offsets[a];
        let hi = knot_offsets[a + 1];
        if hi > knots_concat_view.len() || lo > hi {
            return Err(py_value_error(format!(
                "bspline_tensor_input_location_first_derivative: knot_offsets axis {a} \
                 out of range (lo={lo}, hi={hi}, total={})",
                knots_concat_view.len()
            )));
        }
        per_axis_views.push(knots_concat_view.slice(s![lo..hi]));
    }
    let jet = bspline_tensor_first_derivative(t.as_array(), &per_axis_views, &degrees)
        .map_err(|err| py_value_error(err.to_string()))?;
    Ok(jet.into_pyarray(py).unbind())
}

#[pyfunction(signature = (group, w, g, z, weight, ard_weight, log_bandwidth = None))]
fn equivariant_penalty_value<'py>(
    group: String,
    w: PyReadonlyArray3<'py, f64>,
    g: PyReadonlyArrayDyn<'py, f64>,
    z: PyReadonlyArray2<'py, f64>,
    weight: f64,
    ard_weight: f64,
    log_bandwidth: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<f64> {
    let log_bandwidth = log_bandwidth.as_ref().map(|values| values.as_array());
    gam::terms::equivariant_penalty::equivariant_penalty_value(
        group.as_str(),
        w.as_array(),
        g.as_array(),
        z.as_array(),
        weight,
        ard_weight,
        log_bandwidth,
    )
    .map_err(py_value_error)
}

// ===========================================================================
// Response-geometry transforms (simplex + sphere) and equivariant rho/jvp +
// gauge-companion loss — rustified from gamfit/_response_geometry.py and
// gamfit/_equivariant.py. Each pyfunction is a thin marshalled entrypoint
// that delegates to a pure-Rust impl on ndarray views.
// ===========================================================================

fn rg_vector_norm(v: ArrayView1<'_, f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rg_validate_array(values: ArrayView2<'_, f64>, label: &str) -> Result<(), String> {
    let (n, d) = values.dim();
    if n == 0 || d < 2 {
        return Err(format!(
            "{label} must have at least one row and at least two columns"
        ));
    }
    if let Some(((row, col), value)) = values.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "{label} must contain only finite values; got {value} at ({row}, {col})"
        ));
    }
    Ok(())
}

fn rg_normalize_sphere_matrix(values: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    rg_validate_array(values, "spherical values")?;
    let (n, d) = values.dim();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let norm = rg_vector_norm(values.row(row));
        if norm <= 0.0 {
            return Err("spherical rows must have non-zero norm".to_string());
        }
        for col in 0..d {
            out[[row, col]] = values[[row, col]] / norm;
        }
    }
    Ok(out)
}

fn rg_closure_impl(values: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    rg_validate_array(values, "simplex values")?;
    let (n, d) = values.dim();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut total = 0.0_f64;
        for col in 0..d {
            let v = values[[row, col]];
            if v < 0.0 {
                return Err("simplex values must be non-negative".to_string());
            }
            total += v;
        }
        if total <= 0.0 {
            return Err("simplex rows must have positive total mass".to_string());
        }
        for col in 0..d {
            out[[row, col]] = values[[row, col]] / total;
        }
    }
    Ok(out)
}

fn rg_require_positive(comp: ArrayView2<'_, f64>, label: &str) -> Result<(), String> {
    for value in comp.iter() {
        if *value <= 0.0 {
            return Err(format!("{label} require strictly positive simplex values"));
        }
    }
    Ok(())
}

fn rg_clr_impl(values: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let comp = rg_closure_impl(values)?;
    rg_require_positive(comp.view(), "CLR coordinates")?;
    let (n, d) = comp.dim();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut sum_log = 0.0_f64;
        for col in 0..d {
            let lg = comp[[row, col]].ln();
            out[[row, col]] = lg;
            sum_log += lg;
        }
        let mean = sum_log / (d as f64);
        for col in 0..d {
            out[[row, col]] -= mean;
        }
    }
    Ok(out)
}

fn rg_resolve_reference(reference: isize, d: usize) -> usize {
    let d_i = d as isize;
    let mut r = reference % d_i;
    if r < 0 {
        r += d_i;
    }
    r as usize
}

fn rg_alr_impl(values: ArrayView2<'_, f64>, reference: isize) -> Result<Array2<f64>, String> {
    let comp = rg_closure_impl(values)?;
    rg_require_positive(comp.view(), "ALR coordinates")?;
    let (n, d) = comp.dim();
    let ref_idx = rg_resolve_reference(reference, d);
    let mut out = Array2::<f64>::zeros((n, d - 1));
    for row in 0..n {
        let log_ref = comp[[row, ref_idx]].ln();
        let mut k = 0usize;
        for col in 0..d {
            if col == ref_idx {
                continue;
            }
            out[[row, k]] = comp[[row, col]].ln() - log_ref;
            k += 1;
        }
    }
    Ok(out)
}

fn rg_inverse_alr_impl(
    coords: ArrayView2<'_, f64>,
    reference: isize,
) -> Result<Array2<f64>, String> {
    let (n, dm1) = coords.dim();
    if !coords.iter().all(|v| v.is_finite()) {
        return Err("ALR coordinates must contain only finite values".to_string());
    }
    let d = dm1 + 1;
    let ref_idx = rg_resolve_reference(reference, d);
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut max_v = f64::NEG_INFINITY;
        let mut k = 0usize;
        for col in 0..d {
            let v = if col == ref_idx {
                0.0
            } else {
                let val = coords[[row, k]];
                k += 1;
                val
            };
            out[[row, col]] = v;
            if v > max_v {
                max_v = v;
            }
        }
        let mut total = 0.0_f64;
        for col in 0..d {
            let e = (out[[row, col]] - max_v).exp();
            out[[row, col]] = e;
            total += e;
        }
        for col in 0..d {
            out[[row, col]] /= total;
        }
    }
    Ok(out)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RgSimplexCoord {
    Clr,
    Alr,
}

fn rg_parse_simplex_coord(coordinates: &str) -> Result<RgSimplexCoord, String> {
    match coordinates.to_ascii_lowercase().as_str() {
        "simplex" | "clr" => Ok(RgSimplexCoord::Clr),
        "alr" => Ok(RgSimplexCoord::Alr),
        other => Err(format!(
            "simplex coordinates must be 'clr' or 'alr'; got {other:?}"
        )),
    }
}

fn rg_simplex_log_map_impl(
    values: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
    coord: RgSimplexCoord,
    reference: isize,
) -> Result<Array2<f64>, String> {
    let comp = rg_closure_impl(values)?;
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let base_comp = rg_closure_impl(base2.view())?;
    if comp.ncols() != base_comp.ncols() {
        return Err("simplex values and base point have different dimensions".to_string());
    }
    rg_require_positive(comp.view(), "simplex log map")?;
    rg_require_positive(base_comp.view(), "simplex log map")?;
    match coord {
        RgSimplexCoord::Clr => {
            let values_clr = rg_clr_impl(values)?;
            let base_clr = rg_clr_impl(base2.view())?;
            let (n, d) = values_clr.dim();
            let mut out = Array2::<f64>::zeros((n, d));
            for row in 0..n {
                for col in 0..d {
                    out[[row, col]] = values_clr[[row, col]] - base_clr[[0, col]];
                }
            }
            Ok(out)
        }
        RgSimplexCoord::Alr => {
            let values_alr = rg_alr_impl(values, reference)?;
            let base_alr = rg_alr_impl(base2.view(), reference)?;
            let (n, dm1) = values_alr.dim();
            let mut out = Array2::<f64>::zeros((n, dm1));
            for row in 0..n {
                for col in 0..dm1 {
                    out[[row, col]] = values_alr[[row, col]] - base_alr[[0, col]];
                }
            }
            Ok(out)
        }
    }
}

fn rg_simplex_exp_map_impl(
    tangent: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
    coord: RgSimplexCoord,
    reference: isize,
) -> Result<Array2<f64>, String> {
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let base_comp = rg_closure_impl(base2.view())?;
    let d = base_comp.ncols();
    match coord {
        RgSimplexCoord::Clr => {
            if tangent.ncols() != d {
                return Err("CLR tangent dimension must equal simplex dimension".to_string());
            }
            let n = tangent.nrows();
            let mut out = Array2::<f64>::zeros((n, d));
            for row in 0..n {
                let mut max_v = f64::NEG_INFINITY;
                for col in 0..d {
                    let lg = base_comp[[0, col]].ln() + tangent[[row, col]];
                    out[[row, col]] = lg;
                    if lg > max_v {
                        max_v = lg;
                    }
                }
                let mut total = 0.0_f64;
                for col in 0..d {
                    let e = (out[[row, col]] - max_v).exp();
                    out[[row, col]] = e;
                    total += e;
                }
                for col in 0..d {
                    out[[row, col]] /= total;
                }
            }
            Ok(out)
        }
        RgSimplexCoord::Alr => {
            if tangent.ncols() + 1 != d {
                return Err(
                    "ALR tangent dimension must be simplex dimension minus one".to_string(),
                );
            }
            let base_alr = rg_alr_impl(base2.view(), reference)?;
            let n = tangent.nrows();
            let dm1 = d - 1;
            let mut shifted = Array2::<f64>::zeros((n, dm1));
            for row in 0..n {
                for col in 0..dm1 {
                    shifted[[row, col]] = base_alr[[0, col]] + tangent[[row, col]];
                }
            }
            rg_inverse_alr_impl(shifted.view(), reference)
        }
    }
}

fn rg_sphere_log_map_impl(
    values: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let y = rg_normalize_sphere_matrix(values)?;
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let b_mat = rg_normalize_sphere_matrix(base2.view())?;
    let (n, d) = y.dim();
    if d != b_mat.ncols() {
        return Err("spherical values and base point have different dimensions".to_string());
    }
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut dot = 0.0_f64;
        for col in 0..d {
            dot += y[[row, col]] * b_mat[[0, col]];
        }
        dot = dot.clamp(-1.0, 1.0);
        if dot <= -1.0 + 1.0e-12 {
            return Err("spherical log map is undefined at antipodal points".to_string());
        }
        let theta = dot.acos();
        let sin_theta = theta.sin();
        let scale = if sin_theta > 1.0e-12 {
            theta / sin_theta
        } else {
            1.0
        };
        if theta < 1.0e-12 {
            for col in 0..d {
                out[[row, col]] = 0.0;
            }
        } else {
            for col in 0..d {
                out[[row, col]] = (y[[row, col]] - dot * b_mat[[0, col]]) * scale;
            }
        }
    }
    Ok(out)
}

fn rg_sphere_exp_map_impl(
    tangent: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let b_mat = rg_normalize_sphere_matrix(base2.view())?;
    let (n, d) = tangent.dim();
    if d != b_mat.ncols() {
        return Err("spherical tangent and base point have different dimensions".to_string());
    }
    if !tangent.iter().all(|v| v.is_finite()) {
        return Err("spherical tangent must contain only finite values".to_string());
    }
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut radial = 0.0_f64;
        for col in 0..d {
            radial += tangent[[row, col]] * b_mat[[0, col]];
        }
        let mut z = vec![0.0_f64; d];
        let mut r_sq = 0.0_f64;
        for col in 0..d {
            let v = tangent[[row, col]] - radial * b_mat[[0, col]];
            z[col] = v;
            r_sq += v * v;
        }
        let r = r_sq.sqrt();
        let mut norm_sq = 0.0_f64;
        if r < 1.0e-12 {
            for col in 0..d {
                let v = b_mat[[0, col]] + z[col];
                out[[row, col]] = v;
                norm_sq += v * v;
            }
        } else {
            let cos_r = r.cos();
            let sin_scale = r.sin() / r;
            for col in 0..d {
                let v = cos_r * b_mat[[0, col]] + sin_scale * z[col];
                out[[row, col]] = v;
                norm_sq += v * v;
            }
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err("spherical exponential map produced a non-finite point".to_string());
        }
        for col in 0..d {
            out[[row, col]] /= norm;
        }
    }
    Ok(out)
}

fn rg_normalize_fisher_rao_impl(
    arr: ArrayViewD<'_, f64>,
    n_rows: usize,
    dim: usize,
) -> Result<Array3<f64>, String> {
    if !arr.iter().all(|v| v.is_finite()) {
        return Err("fisher_rao_w must contain only finite values".to_string());
    }
    let shape = arr.shape().to_vec();
    let out: Array3<f64> = match arr.ndim() {
        1 => {
            if shape[0] != n_rows {
                return Err(format!(
                    "fisher_rao_w vector must have length {n_rows}; got {}",
                    shape[0]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                let value = arr[IxDyn(&[row])];
                for d in 0..dim {
                    block[[row, d, d]] = value;
                }
            }
            block
        }
        2 => {
            if shape[0] != dim || shape[1] != dim {
                return Err(format!(
                    "fisher_rao_w matrix must have shape ({dim}, {dim}); got ({}, {})",
                    shape[0], shape[1]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                for r in 0..dim {
                    for c in 0..dim {
                        block[[row, r, c]] = arr[IxDyn(&[r, c])];
                    }
                }
            }
            block
        }
        3 => {
            if shape[0] != n_rows || shape[1] != dim || shape[2] != dim {
                return Err(format!(
                    "fisher_rao_w must have shape ({n_rows}, {dim}, {dim}); got ({}, {}, {})",
                    shape[0], shape[1], shape[2]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                for r in 0..dim {
                    for c in 0..dim {
                        block[[row, r, c]] = arr[IxDyn(&[row, r, c])];
                    }
                }
            }
            block
        }
        _ => return Err("fisher_rao_w must be a 1-D, 2-D, or 3-D numeric array".to_string()),
    };
    for row in 0..n_rows {
        for r in 0..dim {
            for c in 0..dim {
                let a = out[[row, r, c]];
                let b = out[[row, c, r]];
                if (a - b).abs() > 1.0e-10 * (1.0 + a.abs() + b.abs()) {
                    return Err("fisher_rao_w must be symmetric in every row block".to_string());
                }
            }
            if out[[row, r, r]] < 0.0 {
                return Err("fisher_rao_w diagonal entries must be non-negative".to_string());
            }
        }
    }
    Ok(out)
}

fn rg_rho_so2_impl(theta: ArrayView1<'_, f64>) -> Array3<f64> {
    let n = theta.len();
    let mut out = Array3::<f64>::zeros((n, 2, 2));
    for (i, &t) in theta.iter().enumerate() {
        let (s, c) = t.sin_cos();
        out[[i, 0, 0]] = c;
        out[[i, 0, 1]] = -s;
        out[[i, 1, 0]] = s;
        out[[i, 1, 1]] = c;
    }
    out
}

fn rg_rho_so2_jvp_impl(theta: ArrayView1<'_, f64>) -> Array3<f64> {
    let n = theta.len();
    let mut out = Array3::<f64>::zeros((n, 2, 2));
    for (i, &t) in theta.iter().enumerate() {
        let (s, c) = t.sin_cos();
        out[[i, 0, 0]] = -s;
        out[[i, 0, 1]] = -c;
        out[[i, 1, 0]] = c;
        out[[i, 1, 1]] = -s;
    }
    out
}

fn rg_rho_so3_single(ox: f64, oy: f64, oz: f64) -> [[f64; 3]; 3] {
    let angle = (ox * ox + oy * oy + oz * oz).sqrt().max(1.0e-12);
    let ax = ox / angle;
    let ay = oy / angle;
    let az = oz / angle;
    let k = [[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]];
    let mut kk = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for r in 0..3 {
                acc += k[i][r] * k[r][j];
            }
            kk[i][j] = acc;
        }
    }
    let s = angle.sin();
    let one_minus_c = 1.0 - angle.cos();
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            out[i][j] = id + s * k[i][j] + one_minus_c * kk[i][j];
        }
    }
    out
}

fn rg_rho_so3_impl(omega: ArrayView2<'_, f64>) -> Result<Array3<f64>, String> {
    if omega.ncols() != 3 {
        return Err(format!(
            "SO(3) rep input must have shape (N, 3); got {}",
            omega.ncols()
        ));
    }
    let n = omega.nrows();
    let mut out = Array3::<f64>::zeros((n, 3, 3));
    for row in 0..n {
        let r = rg_rho_so3_single(omega[[row, 0]], omega[[row, 1]], omega[[row, 2]]);
        for i in 0..3 {
            for j in 0..3 {
                out[[row, i, j]] = r[i][j];
            }
        }
    }
    Ok(out)
}

fn rg_rho_so3_jvp_impl(
    omega: ArrayView2<'_, f64>,
    domega: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, String> {
    if omega.ncols() != 3 || domega.ncols() != 3 {
        return Err("SO(3) rep JVP requires (N, 3) inputs".to_string());
    }
    if omega.nrows() != domega.nrows() {
        return Err("SO(3) rep JVP omega/domega must agree in row count".to_string());
    }
    let n = omega.nrows();
    let mut out = Array3::<f64>::zeros((n, 3, 3));
    for row in 0..n {
        let rg = rg_rho_so3_single(omega[[row, 0]], omega[[row, 1]], omega[[row, 2]]);
        let sx = domega[[row, 0]];
        let sy = domega[[row, 1]];
        let sz = domega[[row, 2]];
        let kd = [[0.0, -sz, sy], [sz, 0.0, -sx], [-sy, sx, 0.0]];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for r in 0..3 {
                    acc += rg[i][r] * kd[r][j];
                }
                out[[row, i, j]] = acc;
            }
        }
    }
    Ok(out)
}

fn rg_gauge_companion_loss_impl(
    aux_values: ArrayView2<'_, f64>,
    theta: ArrayView2<'_, f64>,
    d_aux: usize,
    weight: f64,
) -> Result<f64, String> {
    if !weight.is_finite() {
        return Err("gauge companion weight must be finite".to_string());
    }
    if aux_values.ncols() < 1 {
        return Err("aux_values must have at least one column".to_string());
    }
    if theta.nrows() != aux_values.nrows() {
        return Err("aux_values and theta must agree in row count".to_string());
    }
    let n = aux_values.nrows();
    let n_f = n as f64;
    let mut terms: Vec<f64> = Vec::new();
    let two_pi = std::f64::consts::TAU;
    let mut term0 = 0.0_f64;
    for row in 0..n {
        let h_rad = aux_values[[row, 0]] * two_pi;
        term0 += 1.0 - (theta[[row, 0]] - h_rad).cos();
    }
    terms.push(term0 / n_f);
    if d_aux >= 2 && theta.ncols() >= 2 && aux_values.ncols() >= 2 {
        let mut term1 = 0.0_f64;
        for row in 0..n {
            let diff = theta[[row, 1]].cos() - (2.0 * aux_values[[row, 1]] - 1.0);
            term1 += diff * diff;
        }
        terms.push(term1 / n_f);
    }
    if d_aux >= 3 && theta.ncols() >= 3 && aux_values.ncols() >= 3 {
        let mut term2 = 0.0_f64;
        for row in 0..n {
            let diff = theta[[row, 2]].cos() - (2.0 * aux_values[[row, 2]] - 1.0);
            term2 += diff * diff;
        }
        terms.push(term2 / n_f);
    }
    let total: f64 = terms.iter().sum();
    Ok(weight * total / (terms.len() as f64))
}

#[pyfunction]
fn response_geometry_closure<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_closure", move || {
        rg_closure_impl(arr.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_clr<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_clr", move || {
        rg_clr_impl(arr.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (values, reference = -1))]
fn response_geometry_alr<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_alr", move || {
        rg_alr_impl(arr.view(), reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (coords, reference = -1))]
fn response_geometry_inverse_alr<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = coords.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_inverse_alr", move || {
        rg_inverse_alr_impl(arr.view(), reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (values, weights = None))]
fn response_geometry_simplex_frechet_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = values.as_array().to_owned();
    let w_owned = weights.as_ref().map(|w| w.as_array().to_owned());
    let out = detach_py_result(py, "response_geometry_simplex_frechet_mean", move || {
        simplex_frechet_mean(arr.view(), w_owned.as_ref().map(|w| w.view()))
    })?;
    Ok(Array1::from_vec(out).into_pyarray(py).unbind())
}

#[pyfunction(signature = (values, base, coordinates, reference = -1))]
fn response_geometry_simplex_log_map<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: String,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_simplex_log_map", move || {
        let coord = rg_parse_simplex_coord(&coordinates)?;
        rg_simplex_log_map_impl(arr.view(), base_owned.view(), coord, reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (tangent, base, coordinates, reference = -1))]
fn response_geometry_simplex_exp_map<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: String,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_simplex_exp_map", move || {
        let coord = rg_parse_simplex_coord(&coordinates)?;
        rg_simplex_exp_map_impl(t_owned.view(), base_owned.view(), coord, reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_sphere_log_map<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_sphere_log_map", move || {
        rg_sphere_log_map_impl(arr.view(), base_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_sphere_exp_map<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_sphere_exp_map", move || {
        rg_sphere_exp_map_impl(t_owned.view(), base_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_normalize_fisher_rao<'py>(
    py: Python<'py>,
    value: PyReadonlyArrayDyn<'py, f64>,
    n_rows: usize,
    dim: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    let arr = value.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_normalize_fisher_rao", move || {
        rg_normalize_fisher_rao_impl(arr.view(), n_rows, dim)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (values, weights = None, tol = 1.0e-12, max_iter = 256))]
fn sphere_frechet_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = values.as_array().to_owned();
    let w_owned = weights.as_ref().map(|w| w.as_array().to_owned());
    let mean = detach_py_result(py, "sphere_frechet_mean", move || {
        gam::geometry::sphere::sphere_frechet_mean(
            arr.view(),
            w_owned.as_ref().map(|w| w.view()),
            tol,
            max_iter,
        )
    })?;
    Ok(Array1::from(mean).into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so2<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let theta_owned = theta.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so2", move || {
        Ok(rg_rho_so2_impl(theta_owned.view()))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so2_jvp<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let theta_owned = theta.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so2_jvp", move || {
        Ok(rg_rho_so2_jvp_impl(theta_owned.view()))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so3<'py>(
    py: Python<'py>,
    omega: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let omega_owned = omega.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so3", move || {
        rg_rho_so3_impl(omega_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so3_jvp<'py>(
    py: Python<'py>,
    omega: PyReadonlyArray2<'py, f64>,
    domega: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let o_owned = omega.as_array().to_owned();
    let d_owned = domega.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so3_jvp", move || {
        rg_rho_so3_jvp_impl(o_owned.view(), d_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (aux_values, theta, d_aux, weight = 1.0))]
fn equivariant_gauge_companion_loss<'py>(
    py: Python<'py>,
    aux_values: PyReadonlyArray2<'py, f64>,
    theta: PyReadonlyArray2<'py, f64>,
    d_aux: usize,
    weight: f64,
) -> PyResult<f64> {
    let aux_owned = aux_values.as_array().to_owned();
    let theta_owned = theta.as_array().to_owned();
    detach_py_result(py, "equivariant_gauge_companion_loss", move || {
        rg_gauge_companion_loss_impl(aux_owned.view(), theta_owned.view(), d_aux, weight)
    })
}

#[pyclass(module = "gam_pyffi._rust", name = "EuclideanManifold")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct EuclideanManifold {
    #[pyo3(get, set)]
    dim: i64,
}

#[pymethods]
impl EuclideanManifold {
    #[new]
    fn new(dim: i64) -> Self {
        Self { dim }
    }

    fn __repr__(&self) -> String {
        format!("EuclideanManifold(dim={})", self.dim)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "euclidean")?;
        out.set_item("dim", self.dim)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "CircleManifold")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct CircleManifold {}

#[pymethods]
impl CircleManifold {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn __repr__(&self) -> String {
        "CircleManifold()".to_owned()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "circle")?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "SphereManifold")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct SphereManifold {
    #[pyo3(get, set)]
    intrinsic_dim: i64,
}

#[pymethods]
impl SphereManifold {
    #[new]
    fn new(intrinsic_dim: i64) -> Self {
        Self { intrinsic_dim }
    }

    fn __repr__(&self) -> String {
        format!("SphereManifold(intrinsic_dim={})", self.intrinsic_dim)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "sphere")?;
        out.set_item("intrinsic_dim", self.intrinsic_dim)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "TorusManifold")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TorusManifold {
    #[pyo3(get, set)]
    dim: i64,
}

#[pymethods]
impl TorusManifold {
    #[new]
    fn new(dim: i64) -> Self {
        Self { dim }
    }

    fn __repr__(&self) -> String {
        format!("TorusManifold(dim={})", self.dim)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "torus")?;
        out.set_item("dim", self.dim)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "GrassmannManifold")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct GrassmannManifold {
    #[pyo3(get, set)]
    k: i64,
    #[pyo3(get, set)]
    n: i64,
}

#[pymethods]
impl GrassmannManifold {
    #[new]
    fn new(k: i64, n: i64) -> Self {
        Self { k, n }
    }

    fn __repr__(&self) -> String {
        format!("GrassmannManifold(k={}, n={})", self.k, self.n)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "grassmann")?;
        out.set_item("k", self.k)?;
        out.set_item("n", self.n)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "StiefelManifold")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct StiefelManifold {
    #[pyo3(get, set)]
    k: i64,
    #[pyo3(get, set)]
    n: i64,
}

#[pymethods]
impl StiefelManifold {
    #[new]
    fn new(k: i64, n: i64) -> Self {
        Self { k, n }
    }

    fn __repr__(&self) -> String {
        format!("StiefelManifold(k={}, n={})", self.k, self.n)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "stiefel")?;
        out.set_item("k", self.k)?;
        out.set_item("n", self.n)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "SpdManifold")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct SpdManifold {
    #[pyo3(get, set)]
    n: i64,
}

#[pymethods]
impl SpdManifold {
    #[new]
    fn new(n: i64) -> Self {
        Self { n }
    }

    fn __repr__(&self) -> String {
        format!("SpdManifold(n={})", self.n)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "spd")?;
        out.set_item("n", self.n)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "ProductManifold")]
struct ProductManifold {
    #[pyo3(get, set)]
    parts: Vec<PyObject>,
}

#[pymethods]
impl ProductManifold {
    #[new]
    #[pyo3(signature = (*parts))]
    fn new(_py: Python<'_>, parts: &Bound<'_, PyTuple>) -> Self {
        Self {
            parts: parts.iter().map(|part| part.clone().unbind()).collect(),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut reprs = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            reprs.push(part.bind(py).repr()?.to_str()?.to_owned());
        }
        let tuple_repr = match reprs.as_slice() {
            [] => "()".to_owned(),
            [only] => format!("({},)", only),
            many => format!("({})", many.join(", ")),
        };
        Ok(format!("ProductManifold(parts={})", tuple_repr))
    }

    fn __eq__(&self, other: &Self, py: Python<'_>) -> PyResult<bool> {
        if self.parts.len() != other.parts.len() {
            return Ok(false);
        }
        for (left, right) in self.parts.iter().zip(other.parts.iter()) {
            if !left.bind(py).eq(right.bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        let parts = PyList::empty(py);
        for part in &self.parts {
            let part_bound = part.bind(py);
            if part_bound.hasattr("to_json")? {
                parts.append(part_bound.getattr("to_json")?.call0()?)?;
            } else {
                parts.append(part_bound)?;
            }
        }
        out.set_item("kind", "product")?;
        out.set_item("parts", parts)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "SparsityPenalty")]
struct SparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    kind: String,
    #[pyo3(get, set)]
    weight: PyObject,
    #[pyo3(get, set)]
    eps: f64,
    #[pyo3(get, set)]
    eps_weight: String,
    #[pyo3(get, set)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl SparsityPenalty {
    #[new]
    #[pyo3(signature = (kind = "smooth_l1", weight = "auto", eps = 1.0e-3, eps_weight = "fixed", *, target = "t", weight_schedule = None))]
    fn new(
        py: Python<'_>,
        kind: String,
        weight: PyObject,
        eps: f64,
        eps_weight: String,
        target: PyObject,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        validate_sparsity_weight(py, weight.bind(py), "SparsityPenalty")?;
        if !matches!(kind.as_str(), "smooth_l1" | "hoyer" | "log") {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.kind must be one of 'smooth_l1' | 'hoyer' | 'log', got {kind:?}"
            )));
        }
        if kind == "hoyer" && eps_weight == "auto" {
            return Err(PyValueError::new_err(
                "SparsityPenalty(kind='hoyer'): Hoyer has no smoothing scale, so eps_weight='auto' is not meaningful.",
            ));
        }
        if eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.eps must be > 0, got {eps}"
            )));
        }
        if !matches!(eps_weight.as_str(), "auto" | "fixed") {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.eps_weight must be 'auto' or 'fixed', got {eps_weight:?}"
            )));
        }
        Ok(Self {
            target,
            kind,
            weight,
            eps,
            eps_weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("sparsity_kind", &self.kind)?;
        payload.set_item("weight", self.weight.bind(py))?;
        payload.set_item("eps", self.eps)?;
        payload.set_item("eps_weight", &self.eps_weight)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SparsityPenalty(kind={}, weight={}, eps={}, eps_weight={}, target={}, weight_schedule={})",
            self.kind.as_str(),
            py_repr(py, self.weight.bind(py))?,
            self.eps,
            self.eps_weight.as_str(),
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn validate_sparsity_weight(py: Python<'_>, weight: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(value) = weight.extract::<String>() {
        if value == "auto" {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "{name}.weight: only 'auto' is accepted as a string, got {value:?}"
        )));
    }
    if let Ok(value) = weight.extract::<f64>() {
        if value > 0.0 {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "{name}.weight must be > 0, got {}",
            py_repr(py, weight)?
        )));
    }
    Err(PyTypeError::new_err(format!(
        "{name}.weight must be 'auto' or a positive float, got {}",
        weight.get_type().name()?
    )))
}

fn target_descriptor(py: Python<'_>, target: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    if target.is_instance_of::<PyString>() || target.extract::<isize>().is_ok() {
        return Ok(target.clone().unbind());
    }
    if let Ok(name) = target.getattr("name") {
        return Ok(name.str()?.into_pyobject(py)?.unbind().into());
    }
    Err(PyTypeError::new_err(
        "analytic penalty target must be a latent block name, latent block index, or object exposing a 'name' attribute",
    ))
}

fn py_repr(_py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.repr()?.extract()
}

#[pyclass(module = "gam_pyffi._rust", name = "ARDPenalty")]
struct ARDPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl ARDPenalty {
    #[new]
    #[pyo3(signature = (*, target = "t", weight_schedule = None))]
    fn new(target: PyObject, weight_schedule: Option<PyObject>) -> Self {
        Self {
            target,
            weight_schedule,
        }
    }

    #[classattr]
    const KIND_TAG: &'static str = "ard";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ARDPenalty(target={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "TopKActivationPenalty")]
struct PyTopKActivationPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    k: i64,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl PyTopKActivationPenalty {
    #[new]
    #[pyo3(signature = (k, weight = 1.0, *, target = "t", weight_schedule = None))]
    fn new(
        k: &Bound<'_, PyAny>,
        weight: f64,
        target: PyObject,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let k = k.call_method0("__index__")?.extract::<i64>()?;
        if k <= 0 {
            return Err(PyValueError::new_err(format!(
                "TopKActivationPenalty.k must be > 0, got {k}"
            )));
        }
        if !weight.is_finite() || weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "TopKActivationPenalty.weight must be finite and > 0, got {weight}"
            )));
        }
        Ok(Self {
            target,
            k,
            weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "topk_activation";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("k", self.k)?;
        payload.set_item("weight", self.weight)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "TopKActivationPenalty(k={}, weight={}, target={}, weight_schedule={})",
            self.k,
            self.weight,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "JumpReLUPenalty")]
struct JumpReLUPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    thresholds: Vec<f64>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl JumpReLUPenalty {
    #[new]
    #[pyo3(signature = (thresholds, weight = 1.0, smoothing_eps = 1.0e-3, *, target = "t", weight_schedule = None))]
    fn new(
        py: Python<'_>,
        thresholds: &Bound<'_, PyAny>,
        weight: f64,
        smoothing_eps: f64,
        target: PyObject,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let numpy = py.import("numpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "float")?;
        let thresholds_array = numpy
            .call_method("asarray", (thresholds,), Some(&kwargs))?
            .call_method1("reshape", (-1,))?;
        let thresholds = thresholds_array
            .call_method0("tolist")?
            .extract::<Vec<f64>>()?;
        if thresholds.is_empty() {
            return Err(PyValueError::new_err(
                "JumpReLUPenalty.thresholds must be non-empty",
            ));
        }
        if thresholds
            .iter()
            .any(|threshold| !threshold.is_finite() || *threshold <= 0.0)
        {
            return Err(PyValueError::new_err(
                "JumpReLUPenalty.thresholds must be finite and > 0",
            ));
        }
        if !weight.is_finite() || weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "JumpReLUPenalty.weight must be finite and > 0, got {weight}"
            )));
        }
        if !smoothing_eps.is_finite() || smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "JumpReLUPenalty.smoothing_eps must be finite and > 0, got {smoothing_eps}"
            )));
        }
        Ok(Self {
            target,
            thresholds,
            weight,
            smoothing_eps,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "jumprelu";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("thresholds", self.thresholds.clone())?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "JumpReLUPenalty(thresholds={:?}, weight={}, smoothing_eps={}, target={}, weight_schedule={})",
            self.thresholds,
            self.weight,
            self.smoothing_eps,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn topk_weight_schedule_descriptor(
    _py: Python<'_>,
    schedule: &Option<PyObject>,
) -> PyResult<Option<PyObject>> {
    let Some(schedule) = schedule else {
        return Ok(None);
    };
    let bound = schedule.bind(_py);
    if bound.hasattr("to_rust_descriptor")? {
        return Ok(Some(
            bound.call_method0("to_rust_descriptor")?.unbind().into(),
        ));
    }
    if let Ok(mapping) = bound.downcast::<PyDict>() {
        return Ok(Some(mapping.copy()?.unbind().into()));
    }
    Err(PyTypeError::new_err(
        "weight_schedule must be ScalarWeightSchedule, a mapping, or None",
    ))
}

fn aux_conditional_prior_float_array<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float")?;
    numpy.call_method("asarray", (value,), Some(&kwargs))
}

fn validate_aux_conditional_prior_lambda(
    lambda_per_row: &Bound<'_, PyAny>,
    weight: f64,
    n_eff: i64,
) -> PyResult<()> {
    if !(weight > 0.0) {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.weight must be > 0, got {weight}"
        )));
    }
    if n_eff <= 0 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.n_eff must be > 0, got {n_eff}"
        )));
    }

    let array = lambda_per_row.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let view = array.as_array();
    let shape = view.shape();
    if view.ndim() != 3 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row must have shape (N, d, d), got ndim={}",
            view.ndim()
        )));
    }

    let n_obs = shape[0];
    let rows = shape[1];
    let cols = shape[2];
    if n_obs as i64 != n_eff {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row first dimension must equal n_eff={n_eff}, got {n_obs}"
        )));
    }
    if rows == 0 || cols == 0 || rows != cols {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row must have square non-empty row matrices, got shape ({n_obs}, {rows}, {cols})"
        )));
    }
    if !view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "AuxConditionalPriorPenalty.lambda_per_row must be finite",
        ));
    }

    let mut max_asym = 0.0_f64;
    for obs in 0..n_obs {
        for row in 0..rows {
            for col in 0..cols {
                let asym = (view[IxDyn(&[obs, row, col])] - view[IxDyn(&[obs, col, row])]).abs();
                if asym > max_asym {
                    max_asym = asym;
                }
            }
        }
    }
    if max_asym >= 1.0e-10 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row matrices must be symmetric within 1e-10; max asymmetry is {max_asym:.3e}"
        )));
    }
    Ok(())
}

#[pyclass(module = "gam_pyffi._rust", name = "AuxConditionalPriorPenalty")]
struct AuxConditionalPriorPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    lambda_per_row: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get, set)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl AuxConditionalPriorPenalty {
    #[new]
    #[pyo3(signature = (lambda_per_row, weight, n_eff, learnable = false, *, target = "t"))]
    fn new(
        py: Python<'_>,
        lambda_per_row: &Bound<'_, PyAny>,
        weight: &Bound<'_, PyAny>,
        n_eff: &Bound<'_, PyAny>,
        learnable: &Bound<'_, PyAny>,
        target: PyObject,
    ) -> PyResult<Self> {
        let builtins = py.import("builtins")?;
        let weight = builtins.getattr("float")?.call1((weight,))?.extract::<f64>()?;
        let n_eff = builtins.getattr("int")?.call1((n_eff,))?.extract::<i64>()?;
        let learnable = learnable.is_truthy()?;
        let lambda_per_row = aux_conditional_prior_float_array(py, lambda_per_row)?;
        validate_aux_conditional_prior_lambda(&lambda_per_row, weight, n_eff)?;
        Ok(Self {
            target,
            lambda_per_row: lambda_per_row.unbind(),
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "aux_conditional_prior";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;

        let array = self
            .lambda_per_row
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let view = array.as_array();
        let flattened = view.iter().copied().collect::<Vec<_>>();
        payload.set_item("lambda_per_row", PyList::new(py, flattened)?)?;
        payload.set_item("lambda_per_row_shape", PyList::new(py, view.shape())?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "AuxConditionalPriorPenalty(lambda_per_row={}, weight={}, n_eff={}, learnable={}, target={}, weight_schedule={})",
            py_repr(py, self.lambda_per_row.bind(py))?,
            self.weight,
            self.n_eff,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "BlockSparsityPenalty")]
struct BlockSparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    groups: Vec<Vec<usize>>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get, set)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl BlockSparsityPenalty {
    #[new]
    #[pyo3(signature = (groups, weight, n_eff, smoothing_eps = 1e-6, learnable = false, *, target = "t"))]
    fn new(
        groups: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: i64,
        smoothing_eps: f64,
        learnable: bool,
        target: PyObject,
    ) -> PyResult<Self> {
        let groups = block_sparsity_coerce_groups(groups)?;
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        if smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }
        Ok(Self {
            target,
            groups,
            weight,
            n_eff,
            smoothing_eps,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "block_sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("groups", self.groups.clone())?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "BlockSparsityPenalty(groups={:?}, weight={}, n_eff={}, smoothing_eps={}, learnable={}, target={}, weight_schedule={})",
            self.groups,
            self.weight,
            self.n_eff,
            self.smoothing_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn block_sparsity_groups_type_error() -> PyErr {
    PyTypeError::new_err("BlockSparsityPenalty.groups must be a sequence of integer sequences")
}

fn block_sparsity_coerce_groups(groups: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
    let group_iter = groups
        .try_iter()
        .map_err(|_| block_sparsity_groups_type_error())?;
    let mut coerced = Vec::new();
    for group in group_iter {
        let group = group.map_err(|_| block_sparsity_groups_type_error())?;
        let axis_iter = group
            .try_iter()
            .map_err(|_| block_sparsity_groups_type_error())?;
        let mut axes = Vec::new();
        for axis in axis_iter {
            let axis = axis.map_err(|_| block_sparsity_groups_type_error())?;
            axes.push(
                axis.call_method0("__index__")
                    .and_then(|value| value.extract::<isize>())
                    .map_err(|_| block_sparsity_groups_type_error())?,
            );
        }
        coerced.push(axes);
    }
    if coerced.is_empty() {
        return Err(PyValueError::new_err(
            "BlockSparsityPenalty.groups must not be empty",
        ));
    }

    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(coerced.len());
    for (group_idx, group) in coerced.into_iter().enumerate() {
        if group.is_empty() {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.groups[{group_idx}] must not be empty"
            )));
        }
        let mut out_group = Vec::with_capacity(group.len());
        for axis in group {
            if axis < 0 {
                return Err(PyValueError::new_err(format!(
                    "BlockSparsityPenalty.groups entries must be non-negative; got {axis}"
                )));
            }
            let axis = axis as usize;
            if !seen.insert(axis) {
                return Err(PyValueError::new_err(format!(
                    "BlockSparsityPenalty.groups axis {axis} appears more than once"
                )));
            }
            out_group.push(axis);
        }
        out.push(out_group);
    }
    let max_axis = seen.iter().next_back().copied().unwrap_or(0);
    for axis in 0..=max_axis {
        if !seen.contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.groups must partition contiguous axes from 0; missing axis {axis}"
            )));
        }
    }
    Ok(out)
}

#[pyclass(module = "gam_pyffi._rust", name = "SoftmaxAssignmentSparsityPenalty")]
struct SoftmaxAssignmentSparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    k_atoms: i64,
    #[pyo3(get, set)]
    temperature: f64,
    #[pyo3(get, set)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl SoftmaxAssignmentSparsityPenalty {
    #[new]
    #[pyo3(signature = (k_atoms, temperature = 1.0, *, target = "t", weight_schedule = None))]
    fn new(
        py: Python<'_>,
        k_atoms: &Bound<'_, PyAny>,
        temperature: &Bound<'_, PyAny>,
        target: PyObject,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let builtins = PyModule::import(py, "builtins")?;
        let k_atoms = builtins
            .getattr("int")?
            .call1((k_atoms,))?
            .extract::<i64>()?;
        let temperature = builtins
            .getattr("float")?
            .call1((temperature,))?
            .extract::<f64>()?;
        if k_atoms <= 0 {
            return Err(PyValueError::new_err(format!(
                "SoftmaxAssignmentSparsityPenalty.k_atoms must be > 0, got {k_atoms}"
            )));
        }
        if temperature <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "SoftmaxAssignmentSparsityPenalty.temperature must be > 0, got {temperature}"
            )));
        }
        Ok(Self {
            target,
            k_atoms,
            temperature,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "softmax_assignment_sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("k_atoms", self.k_atoms)?;
        payload.set_item("temperature", self.temperature)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn _to_rust_payload(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_rust_descriptor(py)
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SoftmaxAssignmentSparsityPenalty(target={}, k_atoms={}, temperature={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            self.k_atoms,
            self.temperature,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "IsometryPenalty")]
struct IsometryPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: PyObject,
    #[pyo3(get, set)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl IsometryPenalty {
    #[new]
    #[pyo3(signature = (weight = "auto", *, target = "t", weight_schedule = None))]
    fn new(
        py: Python<'_>,
        weight: PyObject,
        target: PyObject,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        validate_sparsity_weight(py, weight.bind(py), "IsometryPenalty")?;
        Ok(Self {
            target,
            weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "isometry";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight.bind(py))?;
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "IsometryPenalty(weight={}, target={}, weight_schedule={})",
            py_repr(py, self.weight.bind(py))?,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
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
    module.add_class::<EuclideanManifold>()?;
    module.add_class::<CircleManifold>()?;
    module.add_class::<SphereManifold>()?;
    module.add_class::<TorusManifold>()?;
    module.add_class::<GrassmannManifold>()?;
    module.add_class::<StiefelManifold>()?;
    module.add_class::<SpdManifold>()?;
    module.add_class::<ProductManifold>()?;
    module.add_function(wrap_pyfunction!(classify_exception_message, module)?)?;
    module.add_function(wrap_pyfunction!(sklearn_fit_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(interpolate_survival_surface, module)?)?;
    module.add_function(wrap_pyfunction!(interpolate_rows, module)?)?;
    module.add_function(wrap_pyfunction!(survival_chunk_iter_collect, module)?)?;
    module.add_function(wrap_pyfunction!(write_survival_csv, module)?)?;
    module.add_function(wrap_pyfunction!(survival_coerce_times, module)?)?;
    module.add_function(wrap_pyfunction!(survival_parameters_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(survival_collect_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(hazard_from_cumulative, module)?)?;
    module.add_function(wrap_pyfunction!(survival_cumulative_from_survival, module)?)?;
    module.add_function(wrap_pyfunction!(survival_block, module)?)?;
    module.add_function(wrap_pyfunction!(survival_ffi_surface, module)?)?;
    module.add_function(wrap_pyfunction!(numeric_matrix_validate, module)?)?;
    module.add_function(wrap_pyfunction!(marginal_slope_clip_probabilities, module)?)?;
    module.add_function(wrap_pyfunction!(
        transformation_normal_z_from_columns,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(column_stack_f64, module)?)?;
    module.add_function(wrap_pyfunction!(
        survival_prediction_payload_from_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(flat_to_matrix_f64, module)?)?;
    module.add_function(wrap_pyfunction!(vec_to_array1_f64, module)?)?;
    module.add_function(wrap_pyfunction!(extract_row_ids, module)?)?;
    module.add_function(wrap_pyfunction!(default_survival_time_grid, module)?)?;
    module.add_function(wrap_pyfunction!(torch_from_fitted, module)?)?;
    module.add_function(wrap_pyfunction!(fit_table, module)?)?;
    module.add_function(wrap_pyfunction!(fit_array, module)?)?;
    module.add_function(wrap_pyfunction!(load_model, module)?)?;
    module.add_function(wrap_pyfunction!(bayes_factor_log_diff, module)?)?;
    module.add_function(wrap_pyfunction!(saved_model_payload_string, module)?)?;
    module.add_function(wrap_pyfunction!(build_extend_group_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(extend_model_with_group, module)?)?;
    module.add_function(wrap_pyfunction!(validate_formula_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        formula_validation_supported_by_python_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(formula_validation_repr_json, module)?)?;
    module.add_function(wrap_pyfunction!(formula_validation_html_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_predict_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(predict_array, module)?)?;
    module.add_function(wrap_pyfunction!(competing_risks_cif, module)?)?;
    module.add_function(wrap_pyfunction!(
        competing_risks_prediction_payload_from_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        competing_risks_cif_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_sample_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(sample_table, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_table, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_table_dense, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_array, module)?)?;
    module.add_function(wrap_pyfunction!(build_difference_smooth_request_json, module)?)?;
    module.add_function(wrap_pyfunction!(bspline_basis, module)?)?;
    module.add_function(wrap_pyfunction!(bspline_basis_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(periodic_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis, module)?)?;
    module.add_function(wrap_pyfunction!(smoothness_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_function_norm_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_operator_penalties, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_chart_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(thin_plate_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(auto_knots_1d, module)?)?;
    module.add_function(wrap_pyfunction!(auto_centers_1d, module)?)?;
    module.add_function(wrap_pyfunction!(_block_diag, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_weighted_ridge_array, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_weighted_ridge_batch, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_score, module)?)?;
    module.add_function(wrap_pyfunction!(skip_transcoder_reml_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(skip_transcoder_select_reml, module)?)?;
    module.add_function(wrap_pyfunction!(
        tierney_kadane_normalized_score,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(torch_smooth_dispatch_key, module)?)?;
    module.add_function(wrap_pyfunction!(assemble_candidate_formula, module)?)?;
    module.add_function(wrap_pyfunction!(ordered_prediction_columns, module)?)?;
    module.add_function(wrap_pyfunction!(rank_topology_candidates, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_score, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_score_raw, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_edf, module)?)?;
    module.add_function(wrap_pyfunction!(compare_reml_fits, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_backward, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_formula_table, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_blocks_forward, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_blocks_backward, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_with_constraints_forward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_with_constraints_backward,
        module
    )?)?;
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
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_latent, module)?)?;
    module.add_function(wrap_pyfunction!(register_analytic_penalties, module)?)?;
    module.add_function(wrap_pyfunction!(analytic_penalty_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_penalty_value, module)?)?;
    module.add_function(wrap_pyfunction!(riemannian_retract, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_frechet_mean, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_closure, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_clr, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_alr, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_inverse_alr, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_simplex_frechet_mean,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(response_geometry_simplex_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_simplex_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_sphere_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_sphere_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_normalize_fisher_rao,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so2, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so2_jvp, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so3, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so3_jvp, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_gauge_companion_loss, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit_ibp, module)?)?;
    module.add_function(wrap_pyfunction!(gated_sae_decode, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_latent_backward, module)?)?;
    module.add_function(wrap_pyfunction!(glm_reml_fit_latent, module)?)?;
    module.add_function(wrap_pyfunction!(glm_reml_fit_latent_backward, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_predict_bands_table, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_eta_bands, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_credible_interval, module)?)?;
    module.add_function(wrap_pyfunction!(apply_inverse_link_array, module)?)?;
    module.add_function(wrap_pyfunction!(summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(summary_payload_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(summary_repr, module)?)?;
    module.add_function(wrap_pyfunction!(summary_html, module)?)?;
    module.add_function(wrap_pyfunction!(coefficient_state_json, module)?)?;
    module.add_function(wrap_pyfunction!(term_blocks_for_model, module)?)?;
    module.add_function(wrap_pyfunction!(difference_smooth_json, module)?)?;
    module.add_function(wrap_pyfunction!(difference_smooth_rows, module)?)?;
    module.add_function(wrap_pyfunction!(
        cross_fit_shared_precision_groups_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(check_json, module)?)?;
    module.add_function(wrap_pyfunction!(check_payload_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(report_html, module)?)?;
    module.add_function(wrap_pyfunction!(compute_residuals, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_prediction_metrics, module)?)?;
    // LatentCoord input-location derivative helpers (one per basis kind).
    module.add_function(wrap_pyfunction!(
        duchon_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        duchon_polynomial_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        matern_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(matern_basis_gradient_streaming, module)?)?;
    module.add_function(wrap_pyfunction!(
        sphere_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        periodic_bspline_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        bspline_tensor_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(mechanism_sparsity_jacobian, module)?)?;
    module.add_function(wrap_pyfunction!(conditional_prior_ivae, module)?)?;
    module.add_class::<IsometryPenalty>()?;
    module.add_class::<SparsityPenalty>()?;
    module.add_class::<PyTopKActivationPenalty>()?;
    module.add_class::<ARDPenalty>()?;
    module.add_class::<AuxConditionalPriorPenalty>()?;
    module.add_class::<BlockSparsityPenalty>()?;
    module.add_class::<BlockOrthogonalityPenalty>()?;
    Ok(())
}

/// Smoothed column-2-norm penalty on a decoder weight matrix W of shape
/// `(d_obs, k_latent)`. Returns `(value, grad)` where `grad` has the same
/// shape as `W`.
#[pyfunction(signature = (weight, epsilon, w))]
fn mechanism_sparsity_jacobian<'py>(
    py: Python<'py>,
    weight: f64,
    epsilon: f64,
    w: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, Py<PyArray2<f64>>)> {
    let pen = gam::identifiability::MechanismSparsityJacobian::new(weight, epsilon)
        .map_err(py_value_error)?;
    let (value, grad) = pen.value_and_grad(w.as_array());
    Ok((value, grad.into_pyarray(py).unbind()))
}

/// iVAE conditional-Gaussian log-prior. Given per-row `(mean, scale)` arrays
/// of shape `(n_rows, latent_dim)`, returns the negative log-prior value and
/// its gradient w.r.t. `t` (same shape as `t`).
#[pyfunction(signature = (weight, t, mean, scale))]
fn conditional_prior_ivae<'py>(
    py: Python<'py>,
    weight: f64,
    t: PyReadonlyArray2<'py, f64>,
    mean: PyReadonlyArray2<'py, f64>,
    scale: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, Py<PyArray2<f64>>)> {
    let pen = gam::identifiability::ConditionalPriorIvae::new(
        mean.as_array().to_owned(),
        scale.as_array().to_owned(),
        weight,
    )
    .map_err(py_value_error)?;
    let (value, grad) = pen.value_and_grad(t.as_array());
    Ok((value, grad.into_pyarray(py).unbind()))
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

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn py_panic_error(context: &'static str, payload: Box<dyn std::any::Any + Send>) -> PyErr {
    py_value_error(format!(
        "{context} panicked inside Rust boundary: {}",
        panic_payload_message(payload)
    ))
}

fn detach_py_result<T, F>(py: Python<'_>, context: &'static str, f: F) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    match py.detach(move || catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(message)) => Err(py_value_error(message)),
        Err(payload) => Err(py_panic_error(context, payload)),
    }
}

fn detach_estimation_result<T, F>(py: Python<'_>, context: &'static str, f: F) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, EstimationError> + Send + 'static,
{
    detach_py_result(py, context, move || f().map_err(|err| err.to_string()))
}

fn fit_table_impl(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<&str>,
    fisher_rao_w: Option<ArrayView3<'_, f64>>,
) -> Result<Vec<u8>, String> {
    let dataset = dataset_with_inferred_schema(headers, rows)?;
    fit_dataset_impl(dataset, formula, config_json, fisher_rao_w)
}

fn fit_array_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    formula: String,
    config_json: Option<&str>,
    fisher_rao_w: Option<ArrayView3<'_, f64>>,
) -> Result<Vec<u8>, String> {
    let dataset = dataset_from_xy_arrays(x, y, &formula)?;
    fit_dataset_impl(dataset, formula, config_json, fisher_rao_w)
}

fn inject_scalar_fisher_rao_weight(
    dataset: &mut EncodedDataset,
    fit_config: &mut FitConfig,
    fisher_rao_w: ArrayView3<'_, f64>,
) -> Result<(), String> {
    let n = dataset.values.nrows();
    if fisher_rao_w.shape() != [n, 1, 1] {
        return Err(format!(
            "fit_table fisher_rao_w requires scalar blocks of shape ({n}, 1, 1); got {:?}",
            fisher_rao_w.shape()
        ));
    }
    let weight_name = "__gamfit_fisher_rao_weight".to_string();
    if dataset.headers.iter().any(|name| name == &weight_name) {
        return Err(format!(
            "reserved fisher_rao_w column already exists: {weight_name}"
        ));
    }
    let col_map = dataset.column_map();
    let base_weight_col = fit_config
        .weight_column
        .as_ref()
        .map(|name| {
            col_map
                .get(name)
                .copied()
                .ok_or_else(|| format!("weights column {name:?} is not present"))
        })
        .transpose()?;
    let mut combined = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut value = fisher_rao_w[[row, 0, 0]];
        if let Some(col) = base_weight_col {
            value *= dataset.values[[row, col]];
        }
        if !(value.is_finite() && value >= 0.0) {
            return Err(format!(
                "fisher_rao_w scalar row {row} produced invalid non-negative weight {value}"
            ));
        }
        combined[row] = value;
    }
    let old_cols = dataset.values.ncols();
    let mut values = Array2::<f64>::zeros((n, old_cols + 1));
    values.slice_mut(s![.., ..old_cols]).assign(&dataset.values);
    values.column_mut(old_cols).assign(&combined);
    dataset.values = values;
    dataset.headers.push(weight_name.clone());
    dataset.schema.columns.push(SchemaColumn {
        name: weight_name.clone(),
        kind: ColumnKindTag::Continuous,
        levels: Vec::new(),
    });
    dataset.column_kinds.push(ColumnKindTag::Continuous);
    fit_config.weight_column = Some(weight_name);
    Ok(())
}

fn fit_dataset_impl(
    mut dataset: EncodedDataset,
    formula: String,
    config_json: Option<&str>,
    fisher_rao_w: Option<ArrayView3<'_, f64>>,
) -> Result<Vec<u8>, String> {
    // Always-on progress for the Python bindings: every gamfit fit call
    // installs a visualizer session so the [OUTER step] log stream is
    // accompanied by the `Workflow: Fit | step=N | objective=… (best=…,
    // Δ=±…) | |grad|=…` lane and the `Descent: ▇▆▄▃▂▁` sparkline. The
    // session goes through DumbVisualizer in Python land (stdout/stderr
    // are virtually never TTYs under the gamfit import), so output is
    // throttled stderr writes — safe for notebooks and batch scripts.
    // Session is owned by this function so its Drop runs (clearing the
    // static ACTIVE_FEED) before the result is handed back to Python.
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.set_stage("fit", "optimizing penalized likelihood");
    progress.start_workflow_open_ended("Fit");
    let mut fit_config = parse_fit_config(config_json)?;
    if let Some(w) = fisher_rao_w {
        inject_scalar_fisher_rao_weight(&mut dataset, &mut fit_config, w)?;
    }
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let request = materialized.request;

    let mut payload = match request {
        FitRequest::Standard(standard_request) => {
            let family = standard_request.family.clone();
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
                &standard_result.design,
                standard_result.resolvedspec,
                standard_result.adaptive_diagnostics,
                standard_result.wiggle_knots.map(|knots| knots.to_vec()),
                standard_result.wiggle_degree,
            )?
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
    payload.group_metadata = fit_config.group_metadata.clone();
    let model = FittedModel::from_payload(payload);
    serde_json::to_vec(&model).map_err(|err| format!("failed to serialize model: {err}"))
}

fn load_model_impl(model_bytes: &[u8]) -> Result<FittedModel, String> {
    let model: FittedModel = serde_json::from_slice(model_bytes)
        .map_err(|err| format!("failed to parse model json: {err}"))?;
    model.validate_for_persistence()?;
    model.validate_numeric_finiteness()?;
    Ok(model)
}

fn extend_model_with_group_impl(model_bytes: &[u8], request_json: &str) -> Result<Vec<u8>, String> {
    let mut model = load_model_impl(model_bytes)?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "extend_with_group currently supports standard GAM models only; got '{}'",
            prediction_model_class_label(&model)
        ));
    }
    if model.has_link_wiggle() {
        return Err("extend_with_group does not support link-wiggle models".to_string());
    }
    let request: PyExtendGroupRequest = serde_json::from_str(request_json)
        .map_err(|err| format!("failed to parse extend_with_group request: {err}"))?;
    let PyExtendGroupRequest {
        kind,
        name,
        term,
        column,
        level,
        levels,
        metadata,
        prior,
    } = request;
    let kind = kind
        .as_deref()
        .unwrap_or("random-effect-level")
        .replace('_', "-");
    if kind != "random-effect-level" {
        return Err(format!(
            "extend_with_group supports kind='random-effect-level'; got '{kind}'"
        ));
    }
    let mut levels = levels.unwrap_or_default();
    if let Some(level) = level {
        levels.push(level);
    }
    if levels.is_empty() {
        return Err("extend_with_group requires level or levels".to_string());
    }
    let term = match term.or(column) {
        Some(term) => term,
        None => {
            let payload = model.payload();
            let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
                "extend_with_group requires saved resolved_termspec; refit".to_string()
            })?;
            if spec.random_effect_terms.len() == 1 {
                spec.random_effect_terms[0].name.clone()
            } else {
                return Err(
                    "extend_with_group requires term when the model has zero or multiple group terms"
                        .to_string(),
                );
            }
        }
    };

    for level in levels {
        extend_model_with_random_effect_level(
            &mut model,
            term.as_str(),
            name.as_deref(),
            level,
            metadata.clone(),
            prior.clone(),
        )?;
    }
    model.validate_for_persistence()?;
    model.validate_numeric_finiteness()?;
    serde_json::to_vec(&model).map_err(|err| format!("failed to serialize extended model: {err}"))
}

fn extend_model_with_random_effect_level(
    model: &mut FittedModel,
    term_name: &str,
    requested_name: Option<&str>,
    level: serde_json::Value,
    metadata: Option<serde_json::Value>,
    prior: Option<serde_json::Value>,
) -> Result<(), String> {
    let payload: &mut FittedModelPayload = &mut *model;
    let (term_idx, feature_col, penalty_index) = {
        let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
            "extend_with_group requires saved resolved_termspec; refit".to_string()
        })?;
        let term_idx = spec
            .random_effect_terms
            .iter()
            .position(|term| term.name == term_name)
            .ok_or_else(|| format!("extend_with_group unknown random-effect term '{term_name}'"))?;
        (
            term_idx,
            spec.random_effect_terms[term_idx].feature_col,
            random_effect_penalty_index(spec, term_idx),
        )
    };
    let schema = payload
        .data_schema
        .as_mut()
        .ok_or_else(|| "extend_with_group requires saved data_schema; refit".to_string())?;
    let schema_col = schema.columns.get_mut(feature_col).ok_or_else(|| {
        format!(
            "extend_with_group term '{term_name}' feature column {feature_col} out of saved schema bounds"
        )
    })?;
    let (level_bits, encoded_value) = level_bits_for_extension(schema_col, &level)?;
    {
        let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
            "extend_with_group requires saved resolved_termspec; refit".to_string()
        })?;
        let levels = spec.random_effect_terms[term_idx]
            .frozen_levels
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "extend_with_group term '{term_name}' is not frozen; refit with persisted metadata"
                )
            })?;
        if levels.contains(&level_bits) {
            return Err(format!(
                "extend_with_group level {} already exists for random-effect term '{term_name}'",
                compact_json(&level)
            ));
        }
    }
    if payload.deployment_extensions.iter().any(|extension| {
        extension.kind == "random-effect-level"
            && extension.term == term_name
            && extension.level_bits == level_bits
    }) {
        return Err(format!(
            "extend_with_group level {} is already deployed for random-effect term '{term_name}'",
            compact_json(&level)
        ));
    }
    let coefficient_index = payload
        .fit_result
        .as_ref()
        .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?
        .beta
        .len();
    let (coefficient_mean, supplied_variance) = extension_prior_parameters(prior.as_ref())?;
    let coefficient_variance = match supplied_variance {
        Some(variance) => variance,
        None => {
            let lambda = payload
                .fit_result
                .as_ref()
                .and_then(|fit| fit.lambdas.get(penalty_index).copied())
                .filter(|lambda| lambda.is_finite() && *lambda > 0.0)
                .ok_or_else(|| {
                    format!(
                        "extend_with_group term '{term_name}' has no finite positive prior lambda"
                    )
                })?;
            1.0 / lambda
        }
    };
    extend_training_feature_range(
        payload.training_feature_ranges.as_mut(),
        feature_col,
        encoded_value,
    );
    insert_coefficient_into_saved_fit(
        payload.fit_result.as_mut(),
        coefficient_index,
        coefficient_mean,
        coefficient_variance,
    )?;
    insert_coefficient_into_saved_fit(
        payload.unified.as_mut(),
        coefficient_index,
        coefficient_mean,
        coefficient_variance,
    )?;

    let extension_name = requested_name
        .map(str::to_string)
        .unwrap_or_else(|| format!("{term_name}:{}", compact_json(&level)));
    if let Some(metadata_value) = metadata.clone() {
        let group_metadata = payload.group_metadata.get_or_insert_with(BTreeMap::new);
        group_metadata.insert(extension_name.clone(), metadata_value.clone());
    }
    payload
        .deployment_extensions
        .push(SavedDeploymentExtension {
            name: extension_name,
            kind: "random-effect-level".to_string(),
            term: term_name.to_string(),
            level,
            level_bits,
            coefficient_index,
            coefficient_mean,
            coefficient_variance,
            metadata,
            prior,
        });
    Ok(())
}

fn level_bits_for_extension(
    schema_col: &mut SchemaColumn,
    level: &serde_json::Value,
) -> Result<(u64, f64), String> {
    match schema_col.kind {
        ColumnKindTag::Categorical => {
            let label = match level {
                serde_json::Value::String(s) => s.clone(),
                other => compact_json(other),
            };
            if schema_col.levels.iter().any(|existing| existing == &label) {
                return Err(format!(
                    "extend_with_group categorical level '{label}' already exists in column '{}'",
                    schema_col.name
                ));
            }
            let encoded = schema_col.levels.len() as f64;
            schema_col.levels.push(label);
            Ok((encoded.to_bits(), encoded))
        }
        ColumnKindTag::Continuous | ColumnKindTag::Binary => {
            let value = json_level_to_f64(level)?;
            Ok((value.to_bits(), value))
        }
    }
}

fn json_level_to_f64(value: &serde_json::Value) -> Result<f64, String> {
    let out = match value {
        serde_json::Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("extend_with_group level {n} is not representable as f64"))?,
        serde_json::Value::String(s) => s
            .parse::<f64>()
            .map_err(|_| format!("extend_with_group level '{s}' is not numeric"))?,
        other => {
            return Err(format!(
                "extend_with_group numeric random-effect levels must be numbers or numeric strings; got {}",
                compact_json(other)
            ));
        }
    };
    if !out.is_finite() {
        return Err(format!(
            "extend_with_group random-effect level must be finite; got {out}"
        ));
    }
    Ok(out)
}

fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

fn random_effect_penalty_index(spec: &TermCollectionSpec, term_idx: usize) -> usize {
    usize::from(spec.linear_terms.iter().any(|term| term.double_penalty)) + term_idx
}

fn extension_prior_parameters(
    prior: Option<&serde_json::Value>,
) -> Result<(f64, Option<f64>), String> {
    let Some(value) = prior else {
        return Ok((0.0, None));
    };
    if value.is_null() {
        return Ok((0.0, None));
    }
    let parsed: PyExtensionPrior = serde_json::from_value(value.clone())
        .map_err(|err| format!("failed to parse extend_with_group prior: {err}"))?;
    let mean = parsed.mean.or(parsed.mu).unwrap_or(0.0);
    if !mean.is_finite() {
        return Err(format!(
            "extend_with_group prior mean must be finite; got {mean}"
        ));
    }
    let variance = match (parsed.variance, parsed.precision) {
        (Some(variance), _) => {
            if !(variance.is_finite() && variance > 0.0) {
                return Err(format!(
                    "extend_with_group prior variance must be finite and positive; got {variance}"
                ));
            }
            Some(variance)
        }
        (None, Some(precision)) => {
            if !(precision.is_finite() && precision > 0.0) {
                return Err(format!(
                    "extend_with_group prior precision must be finite and positive; got {precision}"
                ));
            }
            Some(1.0 / precision)
        }
        (None, None) => None,
    };
    Ok((mean, variance))
}

fn extend_training_feature_range(
    ranges: Option<&mut Vec<(f64, f64)>>,
    feature_col: usize,
    value: f64,
) {
    if let Some(ranges) = ranges
        && let Some((lo, hi)) = ranges.get_mut(feature_col)
    {
        if value.is_finite() {
            *lo = (*lo).min(value);
            *hi = (*hi).max(value);
        }
    }
}

fn insert_coefficient_into_saved_fit(
    fit: Option<&mut gam::estimate::UnifiedFitResult>,
    index: usize,
    value: f64,
    variance: f64,
) -> Result<(), String> {
    let Some(fit) = fit else {
        return Ok(());
    };
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!(
            "extend_with_group coefficient variance must be finite and positive; got {variance}"
        ));
    }
    if index > fit.beta.len() {
        return Err(format!(
            "extend_with_group coefficient index {index} exceeds fit coefficient length {}",
            fit.beta.len()
        ));
    }
    fit.beta = insert_array1(&fit.beta, index, value);
    let block_idx = fit
        .blocks
        .iter()
        .position(|block| block.role == BlockRole::Mean)
        .unwrap_or(0);
    if block_idx >= fit.blocks.len() {
        return Err("extend_with_group saved fit has no coefficient blocks".to_string());
    }
    if index > fit.blocks[block_idx].beta.len() {
        return Err(format!(
            "extend_with_group coefficient index {index} exceeds mean block length {}",
            fit.blocks[block_idx].beta.len()
        ));
    }
    fit.blocks[block_idx].beta = insert_array1(&fit.blocks[block_idx].beta, index, value);
    // No-refit posterior algebra for a deployment-only block:
    //
    // The fitted posterior precision for the original coefficients is H_old.
    // Extending with a new random-effect coefficient b and no likelihood
    // refit contributes only its Gaussian prior,
    //
    //   -log p(b) = 1/2 (b - mu)' (lambda_new S_new) (b - mu) + const.
    //
    // Since no old likelihood rows or old penalties are recomputed, the joint
    // precision is blockdiag(H_old, lambda_new S_new).  Therefore the
    // conditional covariance is blockdiag(V_old, S_new^{-1}/lambda_new).  The
    // current API extends one iid random-effect coordinate at a time, so
    // S_new = [1] and `variance` is exactly 1/lambda_new, or the caller's
    // supplied scalar prior covariance.
    if let Some(cov) = fit.covariance_conditional.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    if let Some(cov) = fit.covariance_corrected.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    let variance_diag = variance;
    let precision_diag = 1.0 / variance_diag;
    if let Some(inference) = fit.inference.as_mut() {
        // Boundary adapter: `penalized_hessian` is the `UnscaledPrecision`
        // newtype; unwrap for the `insert_symmetric_array2` helper and wrap
        // the result back on assignment.
        inference.penalized_hessian = insert_symmetric_array2(
            inference.penalized_hessian.as_array(),
            index,
            precision_diag,
        )?
        .into();
        if let Some(cov) = inference.beta_covariance.as_mut() {
            // `beta_covariance` is the `PhiScaledCovariance` newtype.
            *cov = insert_symmetric_array2(cov.as_array(), index, variance_diag)?.into();
        }
        if let Some(se) = inference.beta_standard_errors.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
        }
        if let Some(cov) = inference.beta_covariance_corrected.as_mut() {
            *cov = insert_symmetric_array2(cov, index, variance_diag)?;
        }
        if let Some(se) = inference.beta_standard_errors_corrected.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
        }
        if let Some(cov) = inference.beta_covariance_frequentist.as_mut() {
            *cov = insert_symmetric_array2(cov, index, 0.0)?;
        }
        if let Some(influence) = inference.coefficient_influence.as_mut() {
            *influence = insert_symmetric_array2(influence, index, 0.0)?;
        }
        if let Some(correction) = inference.smoothing_correction.as_mut() {
            *correction = insert_symmetric_array2(correction, index, 0.0)?;
        }
        if let Some(qs) = inference.reparam_qs.as_mut() {
            *qs = insert_symmetric_array2(qs, index, 1.0)?;
        }
        if let Some(bias) = inference.bias_correction_beta.as_mut() {
            *bias = insert_array1(bias, index, 0.0);
        }
    }
    if let Some(geometry) = fit.geometry.as_mut() {
        geometry.penalized_hessian =
            insert_symmetric_array2(geometry.penalized_hessian.as_array(), index, precision_diag)?
                .into();
    }
    Ok(())
}

fn insert_array1(values: &Array1<f64>, index: usize, value: f64) -> Array1<f64> {
    let mut out = Vec::<f64>::with_capacity(values.len() + 1);
    out.extend(values.iter().take(index).copied());
    out.push(value);
    out.extend(values.iter().skip(index).copied());
    Array1::from_vec(out)
}

fn insert_symmetric_array2(
    matrix: &Array2<f64>,
    index: usize,
    diagonal: f64,
) -> Result<Array2<f64>, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "extend_with_group expected square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if index > matrix.nrows() {
        return Err(format!(
            "extend_with_group matrix insert index {index} exceeds dimension {}",
            matrix.nrows()
        ));
    }
    let n = matrix.nrows();
    let mut out = Array2::<f64>::zeros((n + 1, n + 1));
    for old_i in 0..n {
        let new_i = if old_i < index { old_i } else { old_i + 1 };
        for old_j in 0..n {
            let new_j = if old_j < index { old_j } else { old_j + 1 };
            out[[new_i, new_j]] = matrix[[old_i, old_j]];
        }
    }
    out[[index, index]] = diagonal;
    Ok(out)
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

fn parse_formula_validation_payload_json(
    payload_json: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let payload: serde_json::Value = serde_json::from_str(payload_json)
        .map_err(|err| format!("failed to parse FormulaValidation payload: {err}"))?;
    match payload {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err("FormulaValidation payload must be a JSON object".to_string()),
    }
}

fn json_payload_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => value
            .as_f64()
            .map(|number| number != 0.0)
            .unwrap_or(true),
        serde_json::Value::String(value) => !value.is_empty(),
        serde_json::Value::Array(value) => !value.is_empty(),
        serde_json::Value::Object(value) => !value.is_empty(),
    }
}

fn python_repr_json_value(value: Option<&serde_json::Value>) -> String {
    match value {
        None | Some(serde_json::Value::Null) => "None".to_string(),
        Some(serde_json::Value::Bool(value)) => {
            if *value {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Some(serde_json::Value::Number(value)) => value.to_string(),
        Some(serde_json::Value::String(value)) => python_repr_string(value),
        Some(value @ serde_json::Value::Array(_)) | Some(value @ serde_json::Value::Object(_)) => {
            python_str_json_value(value)
        }
    }
}

fn python_repr_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => {
                out.push_str(&format!("\\x{:02x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('\'');
    out
}

fn python_str_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::Bool(value) => {
            if *value {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => value.clone(),
        serde_json::Value::Array(values) => {
            let inner = values
                .iter()
                .map(|value| match value {
                    serde_json::Value::String(value) => python_repr_string(value),
                    value => python_str_json_value(value),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
        serde_json::Value::Object(values) => {
            let inner = values
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        python_repr_string(key),
                        match value {
                            serde_json::Value::String(value) => python_repr_string(value),
                            value => python_str_json_value(value),
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
    }
}

fn escape_html(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            ch => out.push(ch),
        }
    }
    out
}

fn predict_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<&str>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let model_class = model.predict_model_class();
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
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
        columns.insert(
            "effective_variance".to_string(),
            prediction
                .eta_standard_error
                .iter()
                .map(|se| se * se)
                .collect(),
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
    let preferred = [
        "eta",
        "mean",
        "effective_se",
        "effective_variance",
        "mean_lower",
        "mean_upper",
    ];
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
    #[serde(skip_serializing_if = "Option::is_none")]
    beta: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_flat: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_n: Option<usize>,
}

#[derive(Deserialize)]
struct DesignMatrixDensePayload {
    x_flat: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
}

fn design_matrix_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
    design_matrix_dataset_impl(&model, dataset)
}

fn design_matrix_payload_to_dense(raw: &str) -> Result<Array2<f64>, String> {
    let payload: DesignMatrixDensePayload = serde_json::from_str(raw)
        .map_err(|err| format!("failed to parse design matrix payload: {err}"))?;
    let expected_len = payload
        .n_rows
        .checked_mul(payload.n_cols)
        .ok_or_else(|| "design matrix payload shape overflow".to_string())?;
    if payload.x_flat.len() != expected_len {
        return Err(format!(
            "design matrix FFI payload shape mismatch: got {} floats, expected {} * {}",
            payload.x_flat.len(),
            payload.n_rows,
            payload.n_cols
        ));
    }
    Array2::from_shape_vec((payload.n_rows, payload.n_cols), payload.x_flat)
        .map_err(|err| format!("failed to reshape design matrix payload: {err}"))
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
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let covariance = fit
        .beta_covariance_corrected()
        .or_else(|| fit.beta_covariance());
    let covariance_n = covariance.map(|c| c.nrows());
    let covariance_flat = covariance.map(|c| c.iter().copied().collect::<Vec<_>>());
    let payload = DesignMatrixPayload {
        x_flat,
        n_rows,
        n_cols,
        family_kind: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        model_class: prediction_model_class_label(model),
        beta: Some(fit.beta.to_vec()),
        covariance_flat,
        covariance_n,
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
    let dense = design
        .design
        .try_to_dense_by_chunks("design_matrix prediction design")?;
    append_deployment_extension_columns(
        model.payload(),
        dataset.values.view(),
        &col_map,
        training_headers,
        dense,
    )
    .map_err(|err| err.to_string())
}

#[derive(Serialize)]
struct PosteriorPredictPayload {
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    model_class: String,
    family_kind: String,
}

fn eta_bands_from_matrix(
    eta: ArrayView2<'_, f64>,
    family_kind: &str,
    level: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    posterior_bands::eta_bands_from_matrix(eta, family_kind, level)
}

fn posterior_credible_interval_impl(
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
    level: f64,
) -> Result<Vec<f64>, String> {
    gam::inference::posterior::credible_interval(&samples_flat, n_draws, n_coeffs, level)
}

fn posterior_eta_bands_impl(
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    family_kind: &str,
    level: f64,
) -> Result<String, String> {
    let payload =
        posterior_bands::posterior_eta_bands(eta_flat, n_draws, n_rows, family_kind, level)?;
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize posterior_eta_bands payload: {err}"))
}

fn posterior_predict_bands_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
    level: f64,
) -> Result<String, String> {
    // Reuse the existing predict pipeline to get an eta matrix, then collapse
    // to per-row bands inside Rust so the predict() path never materializes
    // the (n_draws, n_rows) matrix on the Python side.
    let raw = posterior_predict_table_impl(
        model_bytes,
        headers,
        rows,
        samples_flat,
        n_draws,
        n_coeffs,
    )?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse posterior predict payload: {err}"))?;
    let n_draws_out = parsed
        .get("n_draws")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "posterior predict payload missing n_draws".to_string())?
        as usize;
    let n_rows_out = parsed
        .get("n_rows")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "posterior predict payload missing n_rows".to_string())?
        as usize;
    let family_kind = parsed
        .get("family_kind")
        .and_then(|v| v.as_str())
        .unwrap_or("identity")
        .to_string();
    let model_class = parsed
        .get("model_class")
        .and_then(|v| v.as_str())
        .unwrap_or("standard")
        .to_string();
    let eta_flat: Vec<f64> = parsed
        .get("eta_flat")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "posterior predict payload missing eta_flat".to_string())?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();
    let eta = Array2::<f64>::from_shape_vec((n_draws_out, n_rows_out), eta_flat)
        .map_err(|err| format!("failed to reshape eta matrix: {err}"))?;
    let (eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper) =
        eta_bands_from_matrix(eta.view(), &family_kind, level)?;
    let payload = PosteriorPredictBandsPayload {
        eta_mean,
        eta_lower,
        eta_upper,
        mean,
        mean_lower,
        mean_upper,
        n_rows: n_rows_out,
        n_draws: n_draws_out,
        model_class,
        family_kind,
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize posterior_predict_bands payload: {err}"))
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
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
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
    let base_dense = design
        .design
        .try_to_dense_by_chunks("posterior_predict design")?;
    let dense = append_deployment_extension_columns(
        model.payload(),
        dataset.values.view(),
        &col_map,
        training_headers,
        base_dense,
    )?;
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
        family_kind: family_link_kind(&model_likelihood_spec(&model)).to_string(),
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
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
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

/// plain string on the Python side (the Rust `LikelihoodSpec` itself is
/// Returns the inverse-link kind tag emitted to the Python wrapper.
///
/// The tag is intentionally lower-kebab-case so it can be matched as a
/// plain string on the Python side (the Rust `LikelihoodSpec` itself is
/// not part of the FFI surface). Families that don't have a closed-form
/// scalar inverse link (`royston-parmar`, `binomial-sas`,
/// `binomial-beta-logistic`) get their own tags so the Python side can
/// refuse to compute a response-scale prediction by string-compare
/// instead of by guessing.
fn family_link_kind(family: &LikelihoodSpec) -> &'static str {
    match (&family.response, &family.link) {
        (ResponseFamily::RoystonParmar, _) => "royston-parmar",
        (ResponseFamily::Binomial, InverseLink::Sas(_)) => "sas",
        (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => "beta-logistic",
        (ResponseFamily::Binomial, InverseLink::Standard(LinkFunction::Sas)) => "sas",
        (ResponseFamily::Binomial, InverseLink::Standard(LinkFunction::BetaLogistic)) => {
            "beta-logistic"
        }
        _ => match family.link_function() {
            LinkFunction::Identity => "identity",
            LinkFunction::Logit => "logit",
            LinkFunction::Probit => "probit",
            LinkFunction::CLogLog => "cloglog",
            LinkFunction::Log => "log",
            LinkFunction::Sas => "sas",
            LinkFunction::BetaLogistic => "beta-logistic",
        },
    }
}

/// Extract the `LikelihoodSpec` carried by the saved model's `family_state`.
/// Latent-survival / latent-binary states have no GLM-style likelihood; they
/// map to `RoystonParmar` with an identity link for the response-scale tag.
fn model_likelihood_spec(model: &FittedModel) -> LikelihoodSpec {
    match &model.payload().family_state {
        gam::inference::model::FittedFamily::Standard { likelihood, .. }
        | gam::inference::model::FittedFamily::LocationScale { likelihood, .. }
        | gam::inference::model::FittedFamily::MarginalSlope { likelihood, .. }
        | gam::inference::model::FittedFamily::Survival { likelihood, .. }
        | gam::inference::model::FittedFamily::TransformationNormal { likelihood, .. } => {
            likelihood.clone()
        }
        gam::inference::model::FittedFamily::LatentSurvival { .. }
        | gam::inference::model::FittedFamily::LatentBinary { .. } => LikelihoodSpec::new(
            ResponseFamily::RoystonParmar,
            InverseLink::Standard(LinkFunction::Identity),
        ),
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
                None
                | Some("marginal-slope")
                | Some("transformation")
                | Some("weibull")
                | Some("royston-parmar")
                | Some(_) => "nuts",
            }
        }
        PredictModelClass::GaussianLocationScale
        | PredictModelClass::BinomialLocationScale
        | PredictModelClass::BernoulliMarginalSlope
        | PredictModelClass::TransformationNormal => "laplace",
    }
}

fn serialize_sample_payload(
    model: &FittedModel,
    nuts: &NutsResult,
    cfg: &NutsConfig,
) -> Result<String, String> {
    let payload = build_sample_payload(model, nuts, cfg);
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize sample payload: {err}"))
}

fn build_sample_payload(model: &FittedModel, nuts: &NutsResult, cfg: &NutsConfig) -> SamplePayload {
    let n_draws = nuts.samples.nrows();
    let n_coeffs = nuts.samples.ncols();
    // Row-major flatten — ndarray's iter visits row-major already.
    let samples_flat: Vec<f64> = nuts.samples.iter().copied().collect();
    let coefficient_names: Vec<String> = (0..n_coeffs).map(|j| format!("beta_{j}")).collect();
    SamplePayload {
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
        family_kind: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        method: nuts_method_label(model).to_string(),
    }
}

#[derive(Serialize)]
struct CoefficientStatePayload {
    beta: Vec<f64>,
    covariance_flat: Vec<f64>,
    covariance_n: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_corrected_flat: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_corrected_n: Option<usize>,
    schema: Option<DataSchema>,
    training_feature_ranges: Option<Vec<(f64, f64)>>,
    random_column_ranges: Vec<(usize, usize)>,
    coefficient_provenance: Vec<CoefficientProvenancePayload>,
    term_blocks: Vec<TermBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    group_metadata: Option<GroupMetadata>,
}

#[derive(Serialize, Deserialize, Clone)]
struct TermBlock {
    name: String,
    kind: String,
    start: usize,
    end: usize,
}

#[derive(Deserialize)]
struct TermBlocksPayload {
    term_blocks: Vec<TermBlock>,
}

#[derive(Serialize)]
struct CoefficientProvenancePayload {
    index: usize,
    label: String,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    term: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    column: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

fn categorical_level_name_for_bits(
    schema: Option<&DataSchema>,
    column_name: &str,
    bits: u64,
) -> Option<String> {
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return None;
    }
    let idx = value as usize;
    if (idx as f64 - value).abs() > 1e-12 {
        return None;
    }
    schema
        .and_then(|schema| {
            schema
                .columns
                .iter()
                .find(|column| column.name == column_name)
        })
        .and_then(|column| column.levels.get(idx))
        .cloned()
}

fn smooth_basis_kind_label(basis: &gam::smooth::SmoothBasisSpec) -> &'static str {
    use gam::smooth::SmoothBasisSpec as S;
    match basis {
        S::BSpline1D { .. } => "smooth_bspline1d",
        S::TensorBSpline { .. } => "tensor",
        S::ThinPlate { .. } => "thin_plate",
        S::Sphere { .. } => "sphere",
        S::Matern { .. } => "matern",
        S::Duchon { .. } => "duchon",
        S::Pca { .. } => "pca",
        S::FactorSmooth { .. } => "factor_smooth",
        S::BySmooth { .. } => "by_smooth",
        S::ByVariable { .. } => "by_variable",
        S::FactorSumToZero { .. } => "factor_sum_to_zero",
    }
}

/// Try to derive per-smooth-term column ranges by building a synthetic
/// two-row design from the saved training feature ranges. Returns
/// `(name, range)` pairs in global-column coordinates, or `None` if the
/// build fails (e.g. no training ranges available).
fn smooth_term_column_ranges(
    payload: &FittedModelPayload,
    smooth_start: usize,
) -> Option<Vec<(String, std::ops::Range<usize>)>> {
    let spec = payload.resolved_termspec.as_ref()?;
    if spec.smooth_terms.is_empty() {
        return Some(Vec::new());
    }
    let ranges = payload.training_feature_ranges.as_ref()?;
    let schema = payload.data_schema.as_ref()?;
    let ncols = schema.columns.len();
    if ranges.is_empty() || ncols == 0 {
        return None;
    }
    let mut data = Array2::<f64>::zeros((2, ncols));
    for (col, &(lo, hi)) in ranges.iter().take(ncols).enumerate() {
        let (lo, hi) = if lo.is_finite() && hi.is_finite() {
            (lo, hi)
        } else {
            (0.0, 1.0)
        };
        data[[0, col]] = lo;
        data[[1, col]] = hi;
    }
    let design = build_term_collection_design(data.view(), spec).ok()?;
    let mut out = Vec::with_capacity(design.smooth.terms.len());
    for term in &design.smooth.terms {
        let r = term.coeff_range.clone();
        let global = (smooth_start + r.start)..(smooth_start + r.end);
        out.push((term.name.clone(), global));
    }
    Some(out)
}

fn coefficient_provenance_for_state(
    payload: &FittedModelPayload,
    beta_len: usize,
) -> (Vec<CoefficientProvenancePayload>, Vec<TermBlock>) {
    let mut provenance = (0..beta_len)
        .map(|index| CoefficientProvenancePayload {
            index,
            label: "__global__".to_string(),
            source: "global".to_string(),
            term: None,
            column: None,
            level: None,
            metadata: None,
        })
        .collect::<Vec<_>>();
    let mut blocks: Vec<TermBlock> = Vec::new();

    let Some(spec) = payload.resolved_termspec.as_ref() else {
        return (provenance, blocks);
    };

    if !provenance.is_empty() {
        provenance[0].term = Some("intercept".to_string());
        provenance[0].label = "intercept".to_string();
        blocks.push(TermBlock {
            name: "intercept".to_string(),
            kind: "intercept".to_string(),
            start: 0,
            end: 1,
        });
    }

    for (offset, term) in spec.linear_terms.iter().enumerate() {
        let index = 1 + offset;
        if let Some(entry) = provenance.get_mut(index) {
            entry.term = Some(term.name.clone());
            entry.column = Some(term.name.clone());
            entry.label = term.name.clone();
            entry.source = "linear".to_string();
        }
        blocks.push(TermBlock {
            name: term.name.clone(),
            kind: "linear".to_string(),
            start: index,
            end: index + 1,
        });
    }

    let mut col = 1 + spec.linear_terms.len();
    for term in &spec.random_effect_terms {
        let levels = term.frozen_levels.as_deref().unwrap_or(&[]);
        let block_start = col;
        for (local, bits) in levels.iter().copied().enumerate() {
            let index = col + local;
            let label =
                categorical_level_name_for_bits(payload.data_schema.as_ref(), &term.name, bits)
                    .unwrap_or_else(|| f64::from_bits(bits).to_string());
            if let Some(entry) = provenance.get_mut(index) {
                entry.label = label.clone();
                entry.source = "group".to_string();
                entry.term = Some(term.name.clone());
                entry.column = Some(term.name.clone());
                entry.level = Some(label.clone());
                entry.metadata = payload
                    .group_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get(&label))
                    .cloned();
            }
        }
        col += levels.len();
        if col > block_start {
            blocks.push(TermBlock {
                name: term.name.clone(),
                kind: "random_effect".to_string(),
                start: block_start,
                end: col,
            });
        }
    }

    // Smooth terms: derive per-term column widths by building a synthetic
    // design from training_feature_ranges. If that fails (older payloads
    // without saved ranges, or unusual basis variants), the columns simply
    // keep their default `__global__` labels.
    if !spec.smooth_terms.is_empty() {
        let smooth_start = col;
        if let Some(smooth_ranges) = smooth_term_column_ranges(payload, smooth_start) {
            for ((name, range), term_spec) in smooth_ranges.iter().zip(spec.smooth_terms.iter()) {
                let kind = smooth_basis_kind_label(&term_spec.basis);
                for idx in range.clone() {
                    if let Some(entry) = provenance.get_mut(idx) {
                        entry.term = Some(name.clone());
                        entry.column = Some(name.clone());
                        entry.label = format!("{}[{}]", name, idx - range.start);
                        entry.source = "smooth".to_string();
                    }
                }
                blocks.push(TermBlock {
                    name: name.clone(),
                    kind: kind.to_string(),
                    start: range.start,
                    end: range.end,
                });
            }
        }
    }

    blocks.sort_by_key(|block| block.start);
    (provenance, blocks)
}

fn coefficient_state_json_impl(model_bytes: &[u8]) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let payload = model.payload();
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let cov = fit
        .beta_covariance()
        .ok_or_else(|| "model does not contain coefficient covariance; refit with covariance-saving inference enabled".to_string())?;
    let (covariance_corrected_flat, covariance_corrected_n) = match fit.beta_covariance_corrected()
    {
        Some(c) => (Some(c.iter().copied().collect()), Some(c.nrows())),
        None => (None, None),
    };
    let mut random_ranges = Vec::<(usize, usize)>::new();
    if let Some(spec) = payload.resolved_termspec.as_ref() {
        let mut col = 1 + spec.linear_terms.len();
        for re in &spec.random_effect_terms {
            let n = re.frozen_levels.as_ref().map(|v| v.len()).unwrap_or(0);
            random_ranges.push((col, col + n));
            col += n;
        }
    }
    let (coefficient_provenance, term_blocks) =
        coefficient_provenance_for_state(payload, fit.beta.len());
    let out = CoefficientStatePayload {
        beta: fit.beta.to_vec(),
        covariance_flat: cov.iter().copied().collect(),
        covariance_n: cov.nrows(),
        covariance_corrected_flat,
        covariance_corrected_n,
        schema: payload.data_schema.clone(),
        training_feature_ranges: payload.training_feature_ranges.clone(),
        random_column_ranges: random_ranges,
        coefficient_provenance,
        term_blocks,
        group_metadata: payload.group_metadata.clone(),
    };
    serde_json::to_string(&out)
        .map_err(|err| format!("failed to serialize coefficient state: {err}"))
}

fn term_blocks_for_model_impl(
    model_bytes: &[u8],
) -> Result<Vec<(String, String, usize, usize)>, String> {
    let state_json = coefficient_state_json_impl(model_bytes)?;
    let payload: TermBlocksPayload = serde_json::from_str(&state_json)
        .map_err(|err| format!("failed to parse coefficient state json: {err}"))?;
    let mut blocks = Vec::with_capacity(payload.term_blocks.len());
    for (idx, block) in payload.term_blocks.into_iter().enumerate() {
        if block.end < block.start {
            return Err(format!(
                "term block {idx} has invalid range [{}, {})",
                block.start, block.end
            ));
        }
        blocks.push((block.name, block.kind, block.start, block.end));
    }
    Ok(blocks)
}

#[derive(Deserialize)]
struct DifferenceSmoothRequest {
    view: String,
    group: Option<String>,
    pairs: Option<Vec<(serde_json::Value, serde_json::Value)>>,
    n: usize,
    level: f64,
    simultaneous: bool,
    n_sim: usize,
    seed: Option<u64>,
    marginalise_random: bool,
    group_means: bool,
    template: Option<BTreeMap<String, serde_json::Value>>,
}

#[derive(Serialize)]
struct DifferenceSmoothRequestJson<'a> {
    view: &'a str,
    group: Option<String>,
    pairs: Option<Vec<(String, String)>>,
    n: usize,
    level: f64,
    simultaneous: bool,
    n_sim: usize,
    seed: Option<u64>,
    marginalise_random: bool,
    group_means: bool,
    template: Option<HashMap<String, String>>,
}

#[pyfunction]
fn build_difference_smooth_request_json(
    view: &str,
    group: Option<String>,
    pairs: Option<Vec<(String, String)>>,
    n: usize,
    level: f64,
    simultaneous: bool,
    n_sim: usize,
    seed: Option<u64>,
    marginalise_random: bool,
    group_means: bool,
    template: Option<HashMap<String, String>>,
) -> PyResult<String> {
    let payload = DifferenceSmoothRequestJson {
        view,
        group,
        pairs,
        n,
        level,
        simultaneous,
        n_sim,
        seed,
        marginalise_random,
        group_means,
        template,
    };
    serde_json::to_string(&payload).map_err(|err| {
        py_value_error(format!(
            "failed to serialize difference_smooth request json: {err}"
        ))
    })
}

fn json_value_to_row_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(flag) => Some(flag.to_string()),
        other => Some(other.to_string()),
    }
}

fn json_value_to_difference_level(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::Bool(flag) => {
            if *flag {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        serde_json::Value::Number(number) => number.to_string(),
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn difference_schema_columns<'a>(
    state: &'a serde_json::Value,
) -> Result<&'a Vec<serde_json::Value>, String> {
    state
        .get("schema")
        .and_then(|schema| schema.get("columns"))
        .and_then(|columns| columns.as_array())
        .ok_or_else(|| "difference_smooth requires a saved model schema".to_string())
}

fn difference_json_ranges(
    state: &serde_json::Value,
    key: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let Some(raw_ranges) = state.get(key).and_then(|raw| raw.as_array()) else {
        return Ok(Vec::new());
    };
    raw_ranges
        .iter()
        .enumerate()
        .map(|(idx, raw)| {
            let values = raw
                .as_array()
                .ok_or_else(|| format!("{key}[{idx}] must be a two-element array"))?;
            if values.len() != 2 {
                return Err(format!("{key}[{idx}] must be a two-element array"));
            }
            let start = values[0]
                .as_u64()
                .ok_or_else(|| format!("{key}[{idx}][0] must be an unsigned integer"))?
                as usize;
            let end = values[1]
                .as_u64()
                .ok_or_else(|| format!("{key}[{idx}][1] must be an unsigned integer"))?
                as usize;
            Ok((start, end))
        })
        .collect()
}

fn difference_training_ranges(state: &serde_json::Value) -> Vec<(f64, f64)> {
    state
        .get("training_feature_ranges")
        .and_then(|raw| raw.as_array())
        .map(|ranges| {
            ranges
                .iter()
                .map(|raw| {
                    let pair = raw.as_array();
                    let lo = pair
                        .and_then(|values| values.first())
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0);
                    let hi = pair
                        .and_then(|values| values.get(1))
                        .and_then(|value| value.as_f64())
                        .unwrap_or(1.0);
                    (lo, hi)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn difference_smooth_covariance(
    state: &serde_json::Value,
    beta_len: usize,
) -> Result<(Array2<f64>, String, bool), String> {
    let corrected = state.get("covariance_corrected_flat").is_some()
        || state.get("covariance_corrected_n").is_some();
    let (flat_key, n_key, kind) = if corrected {
        (
            "covariance_corrected_flat",
            "covariance_corrected_n",
            "corrected",
        )
    } else {
        ("covariance_flat", "covariance_n", "conditional")
    };
    let cov_n = state
        .get(n_key)
        .and_then(|raw| raw.as_u64())
        .ok_or_else(|| format!("model coefficient state does not include {n_key}"))?
        as usize;
    let cov_flat = json_f64_vec(state, flat_key)?;
    if cov_flat.len() != cov_n * cov_n || beta_len != cov_n {
        return Err("coefficient covariance payload has inconsistent dimensions".to_string());
    }
    let cov = Array2::from_shape_vec((cov_n, cov_n), cov_flat)
        .map_err(|err| format!("failed to reshape coefficient covariance: {err}"))?;
    Ok((cov, kind.to_string(), corrected))
}

fn difference_group_ranges(
    state: &serde_json::Value,
    group: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let Some(blocks) = state.get("term_blocks").and_then(|raw| raw.as_array()) else {
        return Ok(Vec::new());
    };
    let mut ranges = Vec::new();
    for block in blocks {
        let Some(obj) = block.as_object() else {
            continue;
        };
        if obj
            .get("kind")
            .and_then(|raw| raw.as_str())
            .is_some_and(|kind| kind == "random_effect")
            && obj
            .get("name")
            .and_then(|raw| raw.as_str())
            .is_some_and(|name| name == group)
        {
            let start = obj
                .get("start")
                .and_then(|raw| raw.as_u64())
                .ok_or_else(|| "term_blocks entry is missing start".to_string())?
                as usize;
            let end = obj
                .get("end")
                .and_then(|raw| raw.as_u64())
                .ok_or_else(|| "term_blocks entry is missing end".to_string())?
                as usize;
            ranges.push((start, end));
        }
    }
    Ok(ranges)
}

fn zero_design_ranges(x: &mut Array2<f64>, ranges: &[(usize, usize)]) {
    let ncols = x.ncols();
    for &(start, end) in ranges {
        let start = start.min(ncols);
        let end = end.min(ncols);
        if start < end {
            x.slice_mut(s![.., start..end]).fill(0.0);
        }
    }
}

struct SplitMixNormalRng {
    state: u64,
    spare: Option<f64>,
}

impl SplitMixNormalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform_open01(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        ((value as f64) + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn uniform_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        if values.len() <= 1 {
            return;
        }
        for i in (1..values.len()).rev() {
            let j = self.uniform_usize(i + 1);
            values.swap(i, j);
        }
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = self.uniform_open01();
        let u2 = self.uniform_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

fn difference_quantile(mut values: Vec<f64>, level: f64) -> Result<f64, String> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return Err("simultaneous difference_smooth simulation produced no finite draws".to_string());
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if values.len() == 1 {
        return Ok(values[0]);
    }
    let pos = level * ((values.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Ok(values[lo])
    } else {
        let weight = pos - (lo as f64);
        Ok(values[lo] * (1.0 - weight) + values[hi] * weight)
    }
}

fn difference_simultaneous_critical(
    beta: &Array1<f64>,
    cov: &Array2<f64>,
    xd: &Array2<f64>,
    diff: &Array1<f64>,
    se: &Array1<f64>,
    n_sim: usize,
    seed: u64,
    level: f64,
) -> Result<f64, String> {
    if n_sim == 0 {
        return Err("difference_smooth n_sim must be at least 1 when simultaneous=True".to_string());
    }
    let sym = (cov.to_owned() + cov.t()) * 0.5;
    let chol = sym
        .cholesky(Side::Lower)
        .map_err(|err| format!("coefficient covariance Cholesky failed: {err}"))?;
    let lower = chol.lower_triangular();
    let n_coeff = beta.len();
    let n_rows = xd.nrows();
    let mut rng = SplitMixNormalRng::new(seed);
    let mut z = vec![0.0; n_coeff];
    let mut draw = vec![0.0; n_coeff];
    let mut max_devs = Vec::with_capacity(n_sim);
    for _ in 0..n_sim {
        for value in &mut z {
            *value = rng.standard_normal();
        }
        for row in 0..n_coeff {
            let mut value = beta[row];
            for col in 0..=row {
                value += lower[[row, col]] * z[col];
            }
            draw[row] = value;
        }
        let mut max_dev = 0.0_f64;
        for i in 0..n_rows {
            let denom = if se[i] > 0.0 { se[i] } else { f64::INFINITY };
            let mut draw_diff = 0.0;
            for j in 0..n_coeff {
                draw_diff += draw[j] * xd[[i, j]];
            }
            let dev = ((draw_diff - diff[i]) / denom).abs();
            if dev.is_finite() && dev > max_dev {
                max_dev = dev;
            }
        }
        max_devs.push(max_dev);
    }
    difference_quantile(max_devs, level)
}

fn difference_smooth_json_impl(model_bytes: &[u8], request_json: &str) -> Result<String, String> {
    use statrs::distribution::ContinuousCDF;

    let state_json = coefficient_state_json_impl(model_bytes)?;
    let state: serde_json::Value = serde_json::from_str(&state_json)
        .map_err(|err| format!("failed to parse coefficient state json: {err}"))?;
    let request: DifferenceSmoothRequest = serde_json::from_str(request_json)
        .map_err(|err| format!("failed to parse difference_smooth request json: {err}"))?;
    if !(0.0 < request.level && request.level < 1.0) {
        return Err("difference_smooth level must be in (0, 1)".to_string());
    }
    if request.n < 2 {
        return Err("difference_smooth n must be at least 2".to_string());
    }

    let schema_cols = difference_schema_columns(&state)?;
    let headers = schema_cols
        .iter()
        .map(|column| {
            column
                .get("name")
                .and_then(|raw| raw.as_str())
                .map(|name| name.to_string())
                .ok_or_else(|| "saved model schema column is missing name".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let view_idx = headers.iter().position(|name| name == &request.view).ok_or_else(|| {
        format!(
            "view column {:?} not found in model schema: {:?}",
            request.view, headers
        )
    })?;
    let group = match request.group {
        Some(group) => group,
        None => schema_cols
            .iter()
            .find_map(|column| {
                let name = column.get("name").and_then(|raw| raw.as_str())?;
                let kind = column.get("kind").and_then(|raw| raw.as_str())?;
                (kind == "categorical" && name != request.view).then(|| name.to_string())
            })
            .ok_or_else(|| {
                "difference_smooth could not infer a categorical group column; pass group="
                    .to_string()
            })?,
    };
    let group_idx = headers.iter().position(|name| name == &group).ok_or_else(|| {
        format!(
            "group column {:?} not found in model schema: {:?}",
            group, headers
        )
    })?;
    let group_col = &schema_cols[group_idx];
    let levels = group_col
        .get("levels")
        .and_then(|raw| raw.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(json_value_to_row_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if levels.len() < 2 {
        return Err(format!(
            "group column {group:?} must have at least two saved levels"
        ));
    }
    let pairs = match request.pairs {
        Some(pairs) => pairs
            .into_iter()
            .map(|(left, right)| {
                (
                    json_value_to_difference_level(&left),
                    json_value_to_difference_level(&right),
                )
            })
            .collect::<Vec<_>>(),
        None => {
            let mut out = Vec::new();
            for i in 0..levels.len() {
                for j in (i + 1)..levels.len() {
                    out.push((levels[i].clone(), levels[j].clone()));
                }
            }
            out
        }
    };

    let ranges = difference_training_ranges(&state);
    let (mut lo, mut hi) = ranges.get(view_idx).copied().unwrap_or((0.0, 1.0));
    if !(lo.is_finite() && hi.is_finite()) || lo == hi {
        lo = 0.0;
        hi = 1.0;
    }
    let step = (hi - lo) / ((request.n - 1) as f64);
    let grid = (0..request.n)
        .map(|idx| lo + step * (idx as f64))
        .collect::<Vec<_>>();

    let mut template = request
        .template
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(key, value)| json_value_to_row_string(&value).map(|text| (key, text)))
        .collect::<BTreeMap<_, _>>();
    for (idx, column) in schema_cols.iter().enumerate() {
        let name = &headers[idx];
        if template.contains_key(name) {
            continue;
        }
        if column
            .get("kind")
            .and_then(|raw| raw.as_str())
            .is_some_and(|kind| kind == "categorical")
        {
            let value = column
                .get("levels")
                .and_then(|raw| raw.as_array())
                .and_then(|values| values.first())
                .and_then(json_value_to_row_string)
                .unwrap_or_else(|| "0".to_string());
            template.insert(name.clone(), value);
        } else {
            let value = ranges
                .get(idx)
                .map(|(a, b)| 0.5 * (a + b))
                .filter(|value| value.is_finite())
                .unwrap_or(0.0);
            template.insert(name.clone(), value.to_string());
        }
    }

    let beta = Array1::from_vec(json_f64_vec(&state, "beta")?);
    let (cov, cov_kind, cov_corrected) = difference_smooth_covariance(&state, beta.len())?;
    let random_ranges = difference_json_ranges(&state, "random_column_ranges")?;
    let group_ranges = difference_group_ranges(&state, &group)?;
    let normal = statrs::distribution::Normal::new(0.0, 1.0)
        .map_err(|err| format!("failed to construct standard normal: {err}"))?;
    let pointwise_crit = normal.inverse_cdf(0.5 + request.level / 2.0);
    let model = load_model_impl(model_bytes)?;
    let mut rows_out = Vec::<serde_json::Value>::new();

    for (level_1, level_2) in pairs {
        let mut rows_left = Vec::with_capacity(grid.len());
        let mut rows_right = Vec::with_capacity(grid.len());
        for x in &grid {
            let mut row_left = template.clone();
            let mut row_right = template.clone();
            row_left.insert(request.view.clone(), x.to_string());
            row_right.insert(request.view.clone(), x.to_string());
            row_left.insert(group.clone(), level_1.clone());
            row_right.insert(group.clone(), level_2.clone());
            rows_left.push(
                headers
                    .iter()
                    .map(|header| row_left.get(header).cloned().unwrap_or_default())
                    .collect::<Vec<_>>(),
            );
            rows_right.push(
                headers
                    .iter()
                    .map(|header| row_right.get(header).cloned().unwrap_or_default())
                    .collect::<Vec<_>>(),
            );
        }
        let dataset_left = dataset_with_model_schema(&model, &headers, &rows_left)?;
        let dataset_right = dataset_with_model_schema(&model, &headers, &rows_right)?;
        let xl = design_matrix_dense(&model, dataset_left)?;
        let xr = design_matrix_dense(&model, dataset_right)?;
        let mut xd = &xr - &xl;
        if request.marginalise_random {
            zero_design_ranges(&mut xd, &random_ranges);
        }
        if !request.group_means {
            zero_design_ranges(&mut xd, &group_ranges);
        }
        let diff = xd.dot(&beta);
        let tmp = xd.dot(&cov);
        let mut var = Array1::<f64>::zeros(xd.nrows());
        for i in 0..xd.nrows() {
            let mut value = 0.0;
            for j in 0..xd.ncols() {
                value += tmp[[i, j]] * xd[[i, j]];
            }
            var[i] = value;
        }
        let se = var.mapv(|value| value.max(0.0).sqrt());
        let crit = if request.simultaneous {
            difference_simultaneous_critical(
                &beta,
                &cov,
                &xd,
                &diff,
                &se,
                request.n_sim,
                request.seed.unwrap_or(12345),
                request.level,
            )?
        } else {
            pointwise_crit
        };
        for (idx, x) in grid.iter().enumerate() {
            let lower = diff[idx] - crit * se[idx];
            let upper = diff[idx] + crit * se[idx];
            let mut row = serde_json::Map::new();
            row.insert(request.view.clone(), serde_json::json!(x));
            row.insert("group".to_string(), serde_json::json!(group.clone()));
            row.insert("level_1".to_string(), serde_json::json!(level_1.clone()));
            row.insert("level_2".to_string(), serde_json::json!(level_2.clone()));
            row.insert("diff".to_string(), serde_json::json!(diff[idx]));
            row.insert("se".to_string(), serde_json::json!(se[idx]));
            row.insert("lower".to_string(), serde_json::json!(lower));
            row.insert("upper".to_string(), serde_json::json!(upper));
            row.insert("level".to_string(), serde_json::json!(request.level));
            row.insert(
                "simultaneous".to_string(),
                serde_json::json!(request.simultaneous),
            );
            row.insert("critical".to_string(), serde_json::json!(crit));
            row.insert(
                "covariance_kind".to_string(),
                serde_json::json!(cov_kind.clone()),
            );
            row.insert(
                "covariance_corrected".to_string(),
                serde_json::json!(cov_corrected),
            );
            rows_out.push(serde_json::Value::Object(row));
        }
    }

    serde_json::to_string(&rows_out)
        .map_err(|err| format!("failed to serialize difference_smooth rows: {err}"))
}

fn json_f64_vec(value: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    let values = value
        .get(key)
        .and_then(|raw| raw.as_array())
        .ok_or_else(|| format!("model coefficient state does not include {key}"))?;
    values
        .iter()
        .enumerate()
        .map(|(idx, raw)| {
            raw.as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("{key}[{idx}] must be finite numeric"))
        })
        .collect()
}

fn json_label_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn coefficient_indices_for_precision_label_json(
    state: &serde_json::Value,
    label: &str,
) -> Result<Vec<usize>, String> {
    let provenance = state
        .get("coefficient_provenance")
        .and_then(|raw| raw.as_array())
        .ok_or_else(|| {
            "model coefficient state does not include coefficient provenance".to_string()
        })?;
    let mut matches = Vec::<usize>::new();
    for (fallback_index, item) in provenance.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let index = obj
            .get("index")
            .and_then(|raw| raw.as_u64())
            .map(|raw| raw as usize)
            .unwrap_or(fallback_index);
        let matched = ["term", "column", "label"].into_iter().any(|key| {
            obj.get(key)
                .and_then(json_label_text)
                .is_some_and(|candidate| candidate == label)
        });
        if matched {
            matches.push(index);
        }
    }
    Ok(matches)
}

fn cross_fit_shared_precision_groups_json_impl(request_json: &str) -> Result<String, String> {
    let request: PySharedPrecisionRequest = serde_json::from_str(request_json)
        .map_err(|err| format!("failed to parse shared precision request json: {err}"))?;
    if request.models.is_empty() {
        return Err("at least one model is required".to_string());
    }
    if request.groups.is_empty() {
        return Err("at least one shared precision group is required".to_string());
    }
    let states = request
        .models
        .iter()
        .map(|model| {
            serde_json::from_str::<serde_json::Value>(&model.state_json).map_err(|err| {
                format!(
                    "failed to parse coefficient state for model {}: {err}",
                    model.key
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut result = serde_json::Map::<String, serde_json::Value>::new();
    for group in request.groups {
        if group.labels.len() != request.models.len() {
            return Err(format!(
                "shared precision group {:?} has {} labels for {} model(s)",
                group.name,
                group.labels.len(),
                request.models.len()
            ));
        }
        if !(group.shape.is_finite() && group.shape > 0.0) {
            return Err(format!(
                "shared precision group {:?} requires finite shape > 0",
                group.name
            ));
        }
        if !(group.rate.is_finite() && group.rate >= 0.0) {
            return Err(format!(
                "shared precision group {:?} requires finite rate >= 0",
                group.name
            ));
        }
        let mut fit_entries = Vec::<serde_json::Value>::new();
        let mut dims = BTreeSet::<usize>::new();
        let mut quadratic_sum = 0.0;
        for ((model, state), label) in request.models.iter().zip(states.iter()).zip(group.labels.iter()) {
            let indices = coefficient_indices_for_precision_label_json(state, label)?;
            if indices.is_empty() {
                continue;
            }
            let beta = json_f64_vec(state, "beta")?;
            let cov_n = state
                .get("covariance_n")
                .and_then(|raw| raw.as_u64())
                .ok_or_else(|| "model coefficient state does not include covariance_n".to_string())?
                as usize;
            let cov_flat = json_f64_vec(state, "covariance_flat")?;
            if beta.len() != cov_n || cov_flat.len() != cov_n * cov_n {
                return Err(format!(
                    "model {} has inconsistent beta/covariance dimensions",
                    model.key
                ));
            }
            if indices.iter().any(|&index| index >= cov_n) {
                return Err(format!(
                    "model {} has coefficient provenance outside covariance bounds",
                    model.key
                ));
            }
            let trace_covariance = indices
                .iter()
                .map(|&index| cov_flat[index * cov_n + index])
                .sum::<f64>();
            let beta_norm_sq = indices
                .iter()
                .map(|&index| beta[index] * beta[index])
                .sum::<f64>();
            let contribution = beta_norm_sq + trace_covariance;
            if !contribution.is_finite() {
                return Err(format!(
                    "shared precision group {:?} has non-finite contribution in model {}",
                    group.name, model.key
                ));
            }
            quadratic_sum += contribution;
            dims.insert(indices.len());
            fit_entries.push(serde_json::json!({
                "model": model.key,
                "label": label,
                "coefficient_indices": indices,
                "dimension": indices.len(),
                "beta_norm_sq": beta_norm_sq,
                "trace_covariance": trace_covariance,
                "quadratic_contribution": contribution,
            }));
        }
        if fit_entries.is_empty() {
            return Err(format!(
                "shared precision group {:?} did not match any model coefficients",
                group.name
            ));
        }
        if dims.len() != 1 {
            return Err(format!(
                "shared precision group {:?} matched inconsistent dimensions: {:?}",
                group.name,
                dims.into_iter().collect::<Vec<_>>()
            ));
        }
        let dimension = *dims.iter().next().expect("dimension checked above");
        let numerator = fit_entries.len() as f64 * dimension as f64 + 2.0 * (group.shape - 1.0);
        let denominator = quadratic_sum + 2.0 * group.rate;
        if numerator <= 0.0 {
            return Err(format!(
                "shared precision group {:?} has non-positive MAP numerator",
                group.name
            ));
        }
        if denominator <= 0.0 || !denominator.is_finite() {
            return Err(format!(
                "shared precision group {:?} has non-positive/non-finite denominator",
                group.name
            ));
        }
        let lambda = numerator / denominator;
        result.insert(
            group.name.clone(),
            serde_json::json!({
                "lambda": lambda,
                "log_lambda": lambda.ln(),
                "shape": group.shape,
                "rate": group.rate,
                "n_fits": fit_entries.len(),
                "dimension": dimension,
                "quadratic_sum": quadratic_sum,
                "numerator": numerator,
                "denominator": denominator,
                "fits": fit_entries,
            }),
        );
    }
    serde_json::to_string(&result)
        .map_err(|err| format!("failed to serialize shared precision result: {err}"))
}

fn summary_json_impl(model_bytes: &[u8]) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let standard_errors = fit
        .beta_standard_errors_corrected()
        .or_else(|| fit.beta_standard_errors());
    let covariance = fit
        .beta_covariance_corrected()
        .map(|cov| ("corrected".to_string(), cov))
        .or_else(|| {
            fit.beta_covariance()
                .map(|cov| ("conditional".to_string(), cov))
        });
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
        family_name: model.likelihood().pretty_name().to_string(),
        model_class: prediction_model_class_label(&model),
        group_metadata: model.payload().group_metadata.clone(),
        deployment_extensions: model.payload().deployment_extensions.clone(),
        deviance: fit.deviance,
        reml_score: fit.reml_score,
        null_space_logdet: fit.artifacts.null_space_logdet,
        null_dim: fit.artifacts.null_space_dim.map(|dim| dim as f64),
        iterations: fit.outer_iterations,
        edf_total: fit.edf_total(),
        lambdas: fit.lambdas.to_vec(),
        coefficients,
        covariance_kind: covariance.as_ref().map(|(kind, _)| kind.clone()),
        covariance_n: covariance.as_ref().map(|(_, cov)| cov.nrows()),
        covariance_flat: covariance.map(|(_, cov)| cov.iter().copied().collect()),
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
                .map(|block| block.role.clone().name().to_string()),
        })
        .collect::<Vec<_>>();
    let report_input = ReportInput {
        model_path: "<in-memory>".to_string(),
        family_name: model.likelihood().pretty_name().to_string(),
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

#[derive(Clone)]
struct LatentPenaltyTarget {
    name: String,
    n: usize,
    d: usize,
}

fn json_u64_to_usize(value: u64, context: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{context} exceeds usize::MAX"))
}

fn json_positive_u64_to_usize(value: u64, context: &str) -> Result<usize, String> {
    if value == 0 {
        return Err(format!("{context} must be > 0"));
    }
    json_u64_to_usize(value, context)
}

fn latent_penalty_targets(
    latents: Option<&serde_json::Value>,
) -> Result<Vec<LatentPenaltyTarget>, String> {
    let Some(raw) = latents.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let map = raw
        .as_object()
        .ok_or_else(|| "latents must be a JSON object keyed by formula symbol".to_string())?;
    let mut out = Vec::with_capacity(map.len());
    for (key, raw_block) in map {
        let obj = raw_block
            .as_object()
            .ok_or_else(|| format!("latents['{key}'] must be an object"))?;
        let name = obj
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(key)
            .to_string();
        let n = json_positive_u64_to_usize(
            obj.get("n")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("latents['{key}'].n is required"))?,
            &format!("latents['{key}'].n"),
        )?;
        let d = json_positive_u64_to_usize(
            obj.get("d")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("latents['{key}'].d is required"))?,
            &format!("latents['{key}'].d"),
        )?;
        out.push(LatentPenaltyTarget { name, n, d });
    }
    Ok(out)
}

fn penalty_target_for_descriptor<'a>(
    targets: &'a [LatentPenaltyTarget],
    descriptor: &serde_json::Map<String, serde_json::Value>,
    context: &str,
) -> Result<&'a LatentPenaltyTarget, String> {
    let raw = descriptor
        .get("target")
        .ok_or_else(|| format!("{context}.target is required"))?;
    if let Some(name) = raw.as_str() {
        return targets
            .iter()
            .find(|target| target.name == name)
            .ok_or_else(|| {
                format!(
                    "{context}.target references latent block {name:?}, but latents declares [{}]",
                    targets
                        .iter()
                        .map(|target| target.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
    }
    if let Some(index) = raw.as_u64() {
        let index = json_u64_to_usize(index, &format!("{context}.target"))?;
        return targets.get(index).ok_or_else(|| {
            format!(
                "{context}.target references latent index {index}, but latents declares {} block(s)",
                targets.len()
            )
        });
    }
    Err(format!(
        "{context}.target must be a latent block name or index"
    ))
}

fn descriptor_f64(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: f64,
) -> Result<f64, String> {
    let value = descriptor
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default);
    if !(value.is_finite() && value > 0.0) {
        return Err(format!("analytic penalty {key} must be finite and > 0"));
    }
    Ok(value)
}

fn descriptor_usize(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> Result<usize, String> {
    let Some(raw) = descriptor.get(key).and_then(serde_json::Value::as_u64) else {
        return Ok(default);
    };
    json_positive_u64_to_usize(raw, &format!("analytic penalty {key}"))
}

fn descriptor_no_unknown_keys(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    context: &str,
    allowed: &[&str],
) -> Result<(), String> {
    for key in descriptor.keys() {
        if !allowed.iter().any(|allowed_key| allowed_key == key) {
            return Err(format!(
                "{context}.{key} is not consumed by the {context} pyffi arm"
            ));
        }
    }
    Ok(())
}

fn descriptor_weight_scalar(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    context: &str,
) -> Result<f64, String> {
    let Some(value) = descriptor.get("weight") else {
        return Ok(1.0);
    };
    if value.as_str() == Some("auto") {
        return Ok(1.0);
    }
    let Some(weight) = value.as_f64() else {
        return Err(format!(
            "{context}.weight must be 'auto' or a finite positive float"
        ));
    };
    if !(weight.is_finite() && weight > 0.0) {
        return Err(format!("{context}.weight must be finite and > 0"));
    }
    Ok(weight)
}

fn descriptor_weight_schedule(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    context: &str,
) -> Result<Option<ScalarWeightSchedule>, String> {
    let Some(raw_schedule) = descriptor.get("weight_schedule") else {
        return Ok(None);
    };
    if raw_schedule.is_null() {
        return Ok(None);
    }
    let schedule = raw_schedule
        .as_object()
        .ok_or_else(|| format!("{context}.weight_schedule must be an object"))?;
    let w_start = schedule
        .get("w_start")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("{context}.weight_schedule.w_start must be a finite number"))?;
    let w_end = schedule
        .get("w_end")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("{context}.weight_schedule.w_end must be a finite number"))?;
    let kind_name = schedule
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("{context}.weight_schedule.kind is required"))?
        .to_ascii_lowercase()
        .replace('-', "_");
    let kind = match kind_name.as_str() {
        "geometric" => {
            let rate = schedule
                .get("rate")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    format!("{context}.weight_schedule.rate is required for geometric")
                })?;
            ScheduleKind::Geometric { rate }
        }
        "linear" => {
            let steps = schedule
                .get("steps")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("{context}.weight_schedule.steps is required for linear"))?;
            ScheduleKind::Linear {
                steps: json_u64_to_usize(steps, &format!("{context}.weight_schedule.steps"))?,
            }
        }
        "reciprocal_iter" => ScheduleKind::ReciprocalIter,
        other => {
            return Err(format!(
                "{context}.weight_schedule.kind must be geometric, linear, or reciprocal_iter; got {other:?}"
            ));
        }
    };
    let mut parsed = ScalarWeightSchedule::new(w_start, w_end, kind)
        .map_err(|err| format!("{context}.weight_schedule: {err}"))?;
    if let Some(iter_count) = schedule.get("iter_count") {
        let raw_iter_count = iter_count.as_u64().ok_or_else(|| {
            format!("{context}.weight_schedule.iter_count must be a non-negative integer")
        })?;
        parsed.iter_count = json_u64_to_usize(
            raw_iter_count,
            &format!("{context}.weight_schedule.iter_count"),
        )?;
    }
    Ok(Some(parsed))
}

fn descriptor_temperature_schedule(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    context: &str,
) -> Result<Option<GumbelTemperatureSchedule>, String> {
    let Some(raw_schedule) = descriptor.get("temperature_schedule") else {
        return Ok(None);
    };
    if raw_schedule.is_null() {
        return Ok(None);
    }
    let schedule = raw_schedule
        .as_object()
        .ok_or_else(|| format!("{context}.temperature_schedule must be an object"))?;
    let tau_start = schedule
        .get("tau_start")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            format!("{context}.temperature_schedule.tau_start must be a finite number")
        })?;
    let tau_min = schedule
        .get("tau_min")
        .or_else(|| schedule.get("tau_end"))
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("{context}.temperature_schedule.tau_min must be a finite number"))?;
    let decay_name = schedule
        .get("decay")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("{context}.temperature_schedule.decay is required"))?
        .to_ascii_lowercase()
        .replace('-', "_");
    let decay = match decay_name.as_str() {
        "geometric" | "exponential" => {
            let rate = schedule
                .get("rate")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.9);
            ScheduleKind::Geometric { rate }
        }
        "linear" => {
            let steps = schedule
                .get("steps")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    format!("{context}.temperature_schedule.steps is required for linear")
                })?;
            ScheduleKind::Linear {
                steps: json_u64_to_usize(steps, &format!("{context}.temperature_schedule.steps"))?,
            }
        }
        "reciprocal_iter" => ScheduleKind::ReciprocalIter,
        other => {
            return Err(format!(
                "{context}.temperature_schedule.decay must be geometric, exponential, linear, or reciprocal_iter; got {other:?}"
            ));
        }
    };
    let mut parsed = GumbelTemperatureSchedule::new(tau_start, tau_min, decay)
        .map_err(|err| format!("{context}.temperature_schedule: {err}"))?;
    if let Some(iter_count) = schedule.get("iter_count") {
        let raw_iter_count = iter_count.as_u64().ok_or_else(|| {
            format!("{context}.temperature_schedule.iter_count must be a non-negative integer")
        })?;
        parsed.iter_count = json_u64_to_usize(
            raw_iter_count,
            &format!("{context}.temperature_schedule.iter_count"),
        )?;
        parsed
            .validate()
            .map_err(|err| format!("{context}.temperature_schedule: {err}"))?;
    }
    Ok(Some(parsed))
}

fn descriptor_difference_op(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    context: &str,
) -> Result<DifferenceOpKind, String> {
    let op = descriptor
        .get("difference_op")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("forward_1d")
        .to_ascii_lowercase()
        .replace('-', "_");
    match op.as_str() {
        "forward_1d" => Ok(DifferenceOpKind::ForwardDiff1D),
        "graph_edges" => {
            let raw_edges = descriptor
                .get("edges")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("{context}.edges is required for graph_edges"))?;
            let mut edges = Vec::with_capacity(raw_edges.len());
            for (edge_idx, raw_edge) in raw_edges.iter().enumerate() {
                let pair = raw_edge.as_array().ok_or_else(|| {
                    format!("{context}.edges[{edge_idx}] must be a two-item list")
                })?;
                if pair.len() != 2 {
                    return Err(format!(
                        "{context}.edges[{edge_idx}] must contain exactly two row indices"
                    ));
                }
                let from = pair[0].as_u64().ok_or_else(|| {
                    format!("{context}.edges[{edge_idx}][0] must be a non-negative integer")
                })?;
                let to = pair[1].as_u64().ok_or_else(|| {
                    format!("{context}.edges[{edge_idx}][1] must be a non-negative integer")
                })?;
                edges.push((
                    json_u64_to_usize(from, &format!("{context}.edges[{edge_idx}][0]"))?,
                    json_u64_to_usize(to, &format!("{context}.edges[{edge_idx}][1]"))?,
                ));
            }
            Ok(DifferenceOpKind::GraphEdges(edges))
        }
        other => Err(format!(
            "{context}.difference_op must be forward_1d or graph_edges; got {other:?}"
        )),
    }
}

fn descriptor_array3_flat(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    data_key: &str,
    shape_key: &str,
    context: &str,
) -> Result<Array3<f64>, String> {
    let shape_values = descriptor
        .get(shape_key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{context}.{shape_key} must be a three-item shape list"))?;
    if shape_values.len() != 3 {
        return Err(format!(
            "{context}.{shape_key} must contain exactly three dimensions"
        ));
    }
    let mut shape = [0usize; 3];
    for (idx, raw_dim) in shape_values.iter().enumerate() {
        let dim = raw_dim
            .as_u64()
            .ok_or_else(|| format!("{context}.{shape_key}[{idx}] must be a positive integer"))?;
        shape[idx] = json_positive_u64_to_usize(dim, &format!("{context}.{shape_key}[{idx}]"))?;
    }
    let expected_len = shape[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_mul(shape[2]))
        .ok_or_else(|| format!("{context}.{shape_key} overflows usize"))?;
    let values = descriptor
        .get(data_key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.len() != expected_len {
        return Err(format!(
            "{context}.{data_key} length {} does not match {shape_key} product {expected_len}",
            values.len()
        ));
    }
    let mut flat = Vec::with_capacity(expected_len);
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), flat)
        .map_err(|err| format!("{context}.{data_key} shape reconstruction failed: {err}"))
}

fn descriptor_array1_flat(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    data_key: &str,
    context: &str,
) -> Result<Array1<f64>, String> {
    let values = descriptor
        .get(data_key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.is_empty() {
        return Err(format!("{context}.{data_key} must be non-empty"));
    }
    let mut flat = Vec::with_capacity(values.len());
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Ok(Array1::from(flat))
}

fn descriptor_array2_flat(
    descriptor: &serde_json::Map<String, serde_json::Value>,
    data_key: &str,
    shape_key: &str,
    context: &str,
) -> Result<Array2<f64>, String> {
    let shape_values = descriptor
        .get(shape_key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{context}.{shape_key} must be a two-item shape list"))?;
    if shape_values.len() != 2 {
        return Err(format!(
            "{context}.{shape_key} must contain exactly two dimensions"
        ));
    }
    let mut shape = [0usize; 2];
    for (idx, raw_dim) in shape_values.iter().enumerate() {
        let dim = raw_dim
            .as_u64()
            .ok_or_else(|| format!("{context}.{shape_key}[{idx}] must be a positive integer"))?;
        shape[idx] = json_positive_u64_to_usize(dim, &format!("{context}.{shape_key}[{idx}]"))?;
    }
    let expected_len = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| format!("{context}.{shape_key} overflows usize"))?;
    let values = descriptor
        .get(data_key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.len() != expected_len {
        return Err(format!(
            "{context}.{data_key} length {} does not match {shape_key} product {expected_len}",
            values.len()
        ));
    }
    let mut flat = Vec::with_capacity(expected_len);
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Array2::from_shape_vec((shape[0], shape[1]), flat)
        .map_err(|err| format!("{context}.{data_key} shape reconstruction failed: {err}"))
}

fn build_analytic_penalty_registry_from_json(
    latents: Option<&serde_json::Value>,
    penalties: Option<&serde_json::Value>,
) -> Result<AnalyticPenaltyRegistry, String> {
    let mut registry = AnalyticPenaltyRegistry::new();
    let Some(raw) = penalties.filter(|value| !value.is_null()) else {
        return Ok(registry);
    };
    let items = raw
        .as_array()
        .ok_or_else(|| "penalties must be a list of analytic penalty descriptors".to_string())?;
    let targets = latent_penalty_targets(latents)?;
    if !items.is_empty() && targets.is_empty() {
        return Err("penalties requires latents with at least one latent block".to_string());
    }
    for (idx, raw_item) in items.iter().enumerate() {
        let context = format!("penalties[{idx}]");
        let descriptor = raw_item
            .as_object()
            .ok_or_else(|| format!("{context} must be an object"))?;
        let target = penalty_target_for_descriptor(&targets, descriptor, &context)?;
        let slice_len = target
            .n
            .checked_mul(target.d)
            .ok_or_else(|| format!("{context}.target latent shape overflows usize"))?;
        let slice = PsiSlice::full(slice_len, Some(target.d));
        let kind = descriptor
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{context}.kind is required"))?
            .to_ascii_lowercase()
            .replace('-', "_");
        let weight_schedule = descriptor_weight_schedule(descriptor, &context)?;
        match kind.as_str() {
            "isometry" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &["kind", "target", "weight", "p_out", "weight_schedule"],
                )?;
                let weight = descriptor_weight_scalar(descriptor, &context)?;
                let p_out = descriptor_usize(descriptor, "p_out", target.d)?;
                let mut penalty = CoreIsometryPenalty::new_euclidean(slice, p_out);
                penalty.scalar_weight = weight;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::Isometry(
                    std::sync::Arc::new(penalty),
                ));
            }
            "ard" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &["kind", "target", "weight_schedule"],
                )?;
                let penalty = CoreARDPenalty::new(slice, target.d);
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::Ard(std::sync::Arc::new(
                    penalty,
                )));
            }
            "topk" | "topk_activation" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &["kind", "target", "k", "weight", "weight_schedule"],
                )?;
                let k = descriptor_usize(descriptor, "k", 1)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let penalty = TopKActivationPenalty::new(slice, k, weight)
                    .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::TopKActivation(
                    std::sync::Arc::new(penalty),
                ));
            }
            "jumprelu" | "jump_relu" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "thresholds",
                        "weight",
                        "smoothing_eps",
                        "weight_schedule",
                    ],
                )?;
                let thresholds = descriptor_array1_flat(descriptor, "thresholds", &context)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-3)?;
                let penalty = RustJumpReLUPenalty::new(slice, thresholds, weight, smoothing_eps)
                    .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::JumpReLU(
                    std::sync::Arc::new(penalty),
                ));
            }
            "orthogonality" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let penalty = CoreOrthogonalityPenalty::new(slice, target.d, weight, n_eff, learnable)
                    .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::Orthogonality(
                    std::sync::Arc::new(penalty),
                ));
            }
            "sparsity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "sparsity_kind",
                        "weight",
                        "eps",
                        "eps_weight",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_weight_scalar(descriptor, &context)?;
                let sparsity_kind = descriptor
                    .get("sparsity_kind")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("smooth_l1")
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let eps = descriptor_f64(descriptor, "eps", 1.0e-3)?;
                let eps_weight = descriptor
                    .get("eps_weight")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("fixed")
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let mut penalty = match sparsity_kind.as_str() {
                    "smooth_l1" | "smoothed_l1" => {
                        CoreSparsityPenalty::smoothed_l1(PenaltyTier::Psi, eps)
                    }
                    "log" => CoreSparsityPenalty::log(PenaltyTier::Psi, eps),
                    "hoyer" => Ok(CoreSparsityPenalty::hoyer(PenaltyTier::Psi)),
                    other => Err(format!(
                        "{context}.sparsity_kind must be smooth_l1, hoyer, or log; got {other:?}"
                    )),
                }?;
                penalty.weight = weight;
                let penalty = match eps_weight.as_str() {
                    "fixed" => penalty,
                    "auto" if sparsity_kind == "hoyer" => {
                        return Err(format!(
                            "{context}.eps_weight='auto' is not meaningful for Hoyer sparsity"
                        ));
                    }
                    "auto" => penalty.with_eps_reml(1),
                    other => {
                        return Err(format!(
                            "{context}.eps_weight must be 'auto' or 'fixed'; got {other:?}"
                        ));
                    }
                };
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::Sparsity(
                    std::sync::Arc::new(penalty),
                ));
            }
            "scad_mcp" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "gamma",
                        "variant",
                        "smoothing_eps",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let variant = descriptor
                    .get("variant")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("mcp")
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let (variant, gamma_default) = match variant.as_str() {
                    "mcp" => (PenaltyConcavity::Mcp, 2.5),
                    "scad" => (PenaltyConcavity::Scad, 3.7),
                    other => {
                        return Err(format!(
                            "{context}.variant must be 'mcp' or 'scad'; got {other:?}"
                        ));
                    }
                };
                let gamma = descriptor_f64(descriptor, "gamma", gamma_default)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let penalty = CoreScadMcpPenalty::new(
                    slice,
                    weight,
                    n_eff,
                    gamma,
                    smoothing_eps,
                    variant,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::ScadMcp(
                    std::sync::Arc::new(penalty),
                ));
            }
            "block_orthogonality" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &["kind", "target", "groups", "weight", "n_eff", "learnable"],
                )?;
                let raw_groups = descriptor
                    .get("groups")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| format!("{context}.groups is required"))?;
                let mut groups = Vec::with_capacity(raw_groups.len());
                for (group_idx, raw_group) in raw_groups.iter().enumerate() {
                    let raw_axes = raw_group.as_array().ok_or_else(|| {
                        format!("{context}.groups[{group_idx}] must be a list of latent axes")
                    })?;
                    let mut group = Vec::with_capacity(raw_axes.len());
                    for (axis_idx, raw_axis) in raw_axes.iter().enumerate() {
                        let raw_axis = raw_axis.as_u64().ok_or_else(|| {
                            format!(
                                "{context}.groups[{group_idx}][{axis_idx}] must be a non-negative integer"
                            )
                        })?;
                        let axis = json_u64_to_usize(
                            raw_axis,
                            &format!("{context}.groups[{group_idx}][{axis_idx}]"),
                        )?;
                        group.push(axis);
                    }
                    groups.push(group);
                }
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                registry.push(gam::terms::AnalyticPenaltyKind::BlockOrthogonality(
                    std::sync::Arc::new(
                        CoreBlockOrthogonalityPenalty::new(slice, groups, weight, n_eff, learnable)
                            .map_err(|err| format!("{context}: {err}"))?,
                    ),
                ));
            }
            "ibp_assignment" | "ibp_assignment_penalty" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "k_max",
                        "alpha",
                        "tau",
                        "learnable",
                        "learnable_alpha",
                        "temperature_schedule",
                        "weight_schedule",
                    ],
                )?;
                let k_max = descriptor_usize(descriptor, "k_max", target.d)?;
                let alpha = descriptor_f64(descriptor, "alpha", 1.0)?;
                let tau = descriptor_f64(descriptor, "tau", 1.0)?;
                let temperature_schedule = descriptor_temperature_schedule(descriptor, &context)?;
                let learnable = descriptor
                    .get("learnable")
                    .or_else(|| descriptor.get("learnable_alpha"))
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let penalty = IBPAssignmentPenalty::new(k_max, alpha, tau, learnable);
                let penalty = match temperature_schedule {
                    Some(schedule) => penalty.with_temperature_schedule(schedule),
                    None => penalty,
                };
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::IBPAssignment(
                    std::sync::Arc::new(penalty),
                ));
            }
            "softmax_assignment_sparsity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "k_atoms",
                        "temperature",
                        "weight_schedule",
                    ],
                )?;
                let k_atoms = descriptor_usize(descriptor, "k_atoms", target.d)?;
                let temperature = descriptor_f64(descriptor, "temperature", 1.0)?;
                let penalty = CoreSoftmaxAssignmentSparsityPenalty::new(k_atoms, temperature);
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::SoftmaxAssignmentSparsity(
                    std::sync::Arc::new(penalty),
                ));
            }
            "total_variation" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "difference_op",
                        "edges",
                        "smoothing_eps",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let difference_op = descriptor_difference_op(descriptor, &context)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let penalty = CoreTotalVariationPenalty::new(
                    weight,
                    n_eff,
                    difference_op,
                    smoothing_eps,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::TotalVariation(
                    std::sync::Arc::new(penalty),
                ));
            }
            "nuclear_norm" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "smoothing_eps",
                        "max_rank",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let max_rank = match descriptor.get("max_rank") {
                    None | Some(serde_json::Value::Null) => None,
                    Some(value) => {
                        let raw = value.as_u64().ok_or_else(|| {
                            format!("{context}.max_rank must be null or a positive integer")
                        })?;
                        Some(json_positive_u64_to_usize(
                            raw,
                            &format!("{context}.max_rank"),
                        )?)
                    }
                };
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let penalty = CoreNuclearNormPenalty::new(
                    slice,
                    weight,
                    n_eff,
                    smoothing_eps,
                    max_rank,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::NuclearNorm(
                    std::sync::Arc::new(penalty),
                ));
            }
            "block_sparsity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "groups",
                        "weight",
                        "n_eff",
                        "smoothing_eps",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let raw_groups = descriptor
                    .get("groups")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| format!("{context}.groups is required"))?;
                let mut groups = Vec::with_capacity(raw_groups.len());
                for (group_idx, raw_group) in raw_groups.iter().enumerate() {
                    let raw_axes = raw_group.as_array().ok_or_else(|| {
                        format!("{context}.groups[{group_idx}] must be a list of latent axes")
                    })?;
                    let mut group = Vec::with_capacity(raw_axes.len());
                    for (axis_idx, raw_axis) in raw_axes.iter().enumerate() {
                        let raw_axis = raw_axis.as_u64().ok_or_else(|| {
                            format!(
                                "{context}.groups[{group_idx}][{axis_idx}] must be a non-negative integer"
                            )
                        })?;
                        let axis = json_u64_to_usize(
                            raw_axis,
                            &format!("{context}.groups[{group_idx}][{axis_idx}]"),
                        )?;
                        group.push(axis);
                    }
                    groups.push(group);
                }
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let penalty = gam::terms::BlockSparsityPenalty::new(
                    slice,
                    groups,
                    weight,
                    n_eff,
                    smoothing_eps,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::BlockSparsity(
                    std::sync::Arc::new(penalty),
                ));
            }
            "mechanism_sparsity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "feature_groups",
                        "weight",
                        "smoothing_eps",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let raw_groups = descriptor
                    .get("feature_groups")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| format!("{context}.feature_groups is required"))?;
                let mut feature_groups = Vec::with_capacity(raw_groups.len());
                let mut max_feature = None::<usize>;
                for (group_idx, raw_group) in raw_groups.iter().enumerate() {
                    let raw_features = raw_group.as_array().ok_or_else(|| {
                        format!("{context}.feature_groups[{group_idx}] must be a list of feature indices")
                    })?;
                    let mut group = Vec::with_capacity(raw_features.len());
                    for (feature_idx, raw_feature) in raw_features.iter().enumerate() {
                        let raw_feature = raw_feature.as_u64().ok_or_else(|| {
                            format!(
                                "{context}.feature_groups[{group_idx}][{feature_idx}] must be a non-negative integer"
                            )
                        })?;
                        let feature = json_u64_to_usize(
                            raw_feature,
                            &format!("{context}.feature_groups[{group_idx}][{feature_idx}]"),
                        )?;
                        max_feature =
                            Some(max_feature.map_or(feature, |current| current.max(feature)));
                        group.push(feature);
                    }
                    feature_groups.push(group);
                }
                let feature_count = max_feature
                    .and_then(|feature| feature.checked_add(1))
                    .ok_or_else(|| format!("{context}.feature_groups must not be empty"))?;
                let target_len = target.d.checked_mul(feature_count).ok_or_else(|| {
                    format!("{context}.target latent_dim × feature_count overflows usize")
                })?;
                let mechanism_slice = PsiSlice::full(target_len, Some(target.d));
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let n_eff = descriptor_f64(descriptor, "n_eff", target.n as f64)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let penalty = MechanismSparsityPenalty::new(
                    mechanism_slice,
                    feature_groups,
                    weight,
                    smoothing_eps,
                    n_eff,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::MechanismSparsity(
                    std::sync::Arc::new(penalty),
                ));
            }
            "aux_conditional_prior" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "lambda_per_row",
                        "lambda_per_row_shape",
                        "weight",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let lambda_per_row = descriptor_array3_flat(
                    descriptor,
                    "lambda_per_row",
                    "lambda_per_row_shape",
                    &context,
                )?;
                let penalty =
                    RowPrecisionPriorPenalty::new(slice, lambda_per_row, weight, n_eff, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::RowPrecisionPrior(
                    std::sync::Arc::new(penalty),
                ));
            }
            "ivae_ridge_mean_gauge" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "aux",
                        "aux_shape",
                        "ridge_eps",
                        "weight",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let aux = descriptor_array2_flat(descriptor, "aux", "aux_shape", &context)?;
                let ridge_eps = descriptor_f64(descriptor, "ridge_eps", 1.0e-6)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let penalty =
                    IvaeRidgeMeanGaugePenalty::new(slice, aux, ridge_eps, weight, n_eff, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(gam::terms::AnalyticPenaltyKind::IvaeRidgeMeanGauge(
                    std::sync::Arc::new(penalty),
                ));
            }
            "parametric_aux_conditional_prior" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "aux",
                        "aux_shape",
                        "log_alpha",
                        "raw_beta",
                        "mu",
                        "mu_shape",
                        "weight",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let aux = descriptor_array2_flat(descriptor, "aux", "aux_shape", &context)?;
                let log_alpha = descriptor_array1_flat(descriptor, "log_alpha", &context)?;
                let raw_beta = descriptor_array1_flat(descriptor, "raw_beta", &context)?;
                let mu = descriptor_array2_flat(descriptor, "mu", "mu_shape", &context)?;
                let penalty = ParametricRowPrecisionPriorPenalty::new(
                    slice, aux, log_alpha, raw_beta, mu, weight, n_eff, learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(
                    gam::terms::AnalyticPenaltyKind::ParametricRowPrecisionPrior(
                        std::sync::Arc::new(penalty),
                    ),
                );
            }
            other => {
                return Err(format!(
                    "{context}.kind has unsupported analytic penalty {other:?}"
                ));
            }
        }
    }
    Ok(registry)
}

#[pyfunction(signature = (latents_json, penalties_json))]
fn register_analytic_penalties(latents_json: &str, penalties_json: &str) -> PyResult<String> {
    let latents: serde_json::Value = serde_json::from_str(latents_json)
        .map_err(|err| py_value_error(format!("invalid latents json: {err}")))?;
    let penalties: serde_json::Value = serde_json::from_str(penalties_json)
        .map_err(|err| py_value_error(format!("invalid penalties json: {err}")))?;
    let registry = build_analytic_penalty_registry_from_json(Some(&latents), Some(&penalties))
        .map_err(py_value_error)?;
    let layout = registry
        .rho_layout()
        .into_iter()
        .map(|(range, tier, name)| {
            serde_json::json!({
                "name": name,
                "tier": format!("{tier:?}"),
                "rho_start": range.start,
                "rho_end": range.end,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&serde_json::json!({
        "penalty_count": registry.penalties.len(),
        "rho_count": registry.total_rho_count(),
        "layout": layout,
    }))
    .map_err(|err| py_value_error(err.to_string()))
}

#[pyfunction(signature = (
    latents_json,
    penalties_json,
    target,
    rho = None,
    isometry_jacobian = None,
    isometry_jacobian_second = None
))]
fn analytic_penalty_value_grad<'py>(
    py: Python<'py>,
    latents_json: &str,
    penalties_json: &str,
    target: PyReadonlyArray1<'py, f64>,
    rho: Option<PyReadonlyArray1<'py, f64>>,
    isometry_jacobian: Option<PyReadonlyArray2<'py, f64>>,
    isometry_jacobian_second: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(f64, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let latents: serde_json::Value = serde_json::from_str(latents_json)
        .map_err(|err| py_value_error(format!("invalid latents json: {err}")))?;
    let penalties: serde_json::Value = serde_json::from_str(penalties_json)
        .map_err(|err| py_value_error(format!("invalid penalties json: {err}")))?;
    let mut registry = build_analytic_penalty_registry_from_json(Some(&latents), Some(&penalties))
        .map_err(py_value_error)?;

    if let Some(j) = isometry_jacobian {
        let j_cache = Arc::new(j.as_array().to_owned());
        let h_cache = isometry_jacobian_second.map(|h| Arc::new(h.as_array().to_owned()));
        for penalty in &mut registry.penalties {
            if let AnalyticPenaltyKind::Isometry(inner) = penalty {
                let mut cloned = (**inner).clone().with_jacobian_cache(j_cache.clone());
                if let Some(h) = h_cache.as_ref() {
                    cloned = cloned.with_jacobian_second_cache(h.clone());
                }
                *penalty = AnalyticPenaltyKind::Isometry(Arc::new(cloned));
            }
        }
    }

    let target_view = target.as_array();
    let rho_owned = match rho {
        Some(rho) => rho.as_array().to_owned(),
        None => Array1::<f64>::zeros(registry.total_rho_count()),
    };
    if rho_owned.len() != registry.total_rho_count() {
        return Err(py_value_error(format!(
            "rho length {} does not match analytic penalty rho_count {}",
            rho_owned.len(),
            registry.total_rho_count()
        )));
    }

    let mut value = 0.0_f64;
    let mut grad = Array1::<f64>::zeros(target_view.len());
    let mut grad_rho = Array1::<f64>::zeros(rho_owned.len());
    for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(registry.rho_layout())
    {
        if matches!(tier, PenaltyTier::Rho) {
            continue;
        }
        let rho_local = rho_owned.slice(s![rho_slice.clone()]);
        value += penalty.value(target_view.view(), rho_local);
        grad += &penalty.grad_target(target_view.view(), rho_local);
        let local_grad_rho = penalty.grad_rho(target_view.view(), rho_local);
        for (local, global) in (rho_slice.start..rho_slice.end).enumerate() {
            grad_rho[global] += local_grad_rho[local];
        }
    }
    Ok((
        value,
        grad.into_pyarray(py).unbind(),
        grad_rho.into_pyarray(py).unbind(),
    ))
}

fn parse_manifold_kind(value: &serde_json::Value) -> Result<gam::geometry::ManifoldSpec, String> {
    if let Some(name) = value.as_str() {
        return match name.to_ascii_lowercase().as_str() {
            "circle" | "s1" => Ok(gam::geometry::ManifoldSpec::Circle),
            other => Err(format!("unknown manifold string {other:?}")),
        };
    }
    let obj = value
        .as_object()
        .ok_or_else(|| "manifold must be a string or object".to_string())?;
    let kind = obj
        .get("kind")
        .or_else(|| obj.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "manifold.kind is required".to_string())?
        .to_ascii_lowercase();
    match kind.as_str() {
        "euclidean" => {
            let dim = obj
                .get("dim")
                .or_else(|| obj.get("d"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "euclidean manifold requires dim".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Euclidean(
                json_positive_u64_to_usize(dim, "euclidean.dim")?,
            ))
        }
        "circle" | "s1" => Ok(gam::geometry::ManifoldSpec::Circle),
        "sphere" => {
            let n = obj
                .get("intrinsic_dim")
                .or_else(|| obj.get("n"))
                .or_else(|| obj.get("dim"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "sphere manifold requires intrinsic_dim".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Sphere {
                intrinsic_dim: json_positive_u64_to_usize(n, "sphere.intrinsic_dim")?,
            })
        }
        "torus" => {
            let d = obj
                .get("d")
                .or_else(|| obj.get("dim"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "torus manifold requires d".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Torus {
                dim: json_positive_u64_to_usize(d, "torus.d")?,
            })
        }
        "grassmann" => {
            let k = obj
                .get("k")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "grassmann manifold requires k".to_string())?;
            let n = obj
                .get("n")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "grassmann manifold requires n".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Grassmann {
                k: json_positive_u64_to_usize(k, "grassmann.k")?,
                n: json_positive_u64_to_usize(n, "grassmann.n")?,
            })
        }
        "stiefel" => {
            let k = obj
                .get("k")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "stiefel manifold requires k".to_string())?;
            let n = obj
                .get("n")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "stiefel manifold requires n".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Stiefel {
                k: json_positive_u64_to_usize(k, "stiefel.k")?,
                n: json_positive_u64_to_usize(n, "stiefel.n")?,
            })
        }
        "spd" => {
            let n = obj
                .get("n")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "spd manifold requires n".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Spd {
                n: json_positive_u64_to_usize(n, "spd.n")?,
            })
        }
        "product" => {
            let parts = obj
                .get("components")
                .or_else(|| obj.get("parts"))
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| "product manifold requires components".to_string())?;
            let mut parsed = Vec::with_capacity(parts.len());
            for part in parts {
                parsed.push(parse_manifold_kind(part)?);
            }
            Ok(gam::geometry::ManifoldSpec::Product(parsed))
        }
        other => Err(format!("unknown manifold kind {other:?}")),
    }
}

#[pyfunction(signature = (manifold_json, points, delta))]
fn riemannian_retract<'py>(
    py: Python<'py>,
    manifold_json: &str,
    points: PyReadonlyArray2<'py, f64>,
    delta: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let value: serde_json::Value = serde_json::from_str(manifold_json)
        .map_err(|err| py_value_error(format!("invalid manifold json: {err}")))?;
    let kind = parse_manifold_kind(&value).map_err(py_value_error)?;
    let manifold = kind.build();
    let p = points.as_array();
    let d = delta.as_array();
    if p.dim() != d.dim() {
        return Err(py_value_error(format!(
            "points shape {:?} does not match delta shape {:?}",
            p.dim(),
            d.dim()
        )));
    }
    if p.ncols() != manifold.ambient_dim() {
        return Err(py_value_error(format!(
            "points width {} does not match manifold ambient_dim {}",
            p.ncols(),
            manifold.ambient_dim()
        )));
    }
    let mut out = Array2::<f64>::zeros(p.dim());
    for row in 0..p.nrows() {
        let tangent = manifold
            .project_tangent(p.row(row), d.row(row))
            .map_err(|err| py_value_error(err.to_string()))?;
        let next = manifold
            .retract(p.row(row), tangent.view())
            .map_err(|err| py_value_error(err.to_string()))?;
        out.row_mut(row).assign(&next);
    }
    Ok(out.into_pyarray(py).unbind())
}

fn parse_fit_config(config_json: Option<&str>) -> Result<FitConfig, String> {
    let py_config = match config_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str::<PyFitConfig>(raw)
            .map_err(|err| format!("invalid fit config json: {err}"))?,
        _ => PyFitConfig::default(),
    };
    let mut fit_config = FitConfig::default();
    fit_config.group_metadata = parse_group_metadata(py_config.group_metadata, py_config.groups)?;
    fit_config.penalty_block_gamma_priors = parse_precision_hyperpriors(
        py_config.precision_hyperpriors,
        py_config.penalty_block_gamma_priors,
    )?;
    let analytic_penalties = py_config.penalties;
    build_analytic_penalty_registry_from_json(
        py_config.latents.as_ref(),
        analytic_penalties.as_ref(),
    )?;
    fit_config.latents = py_config.latents;
    fit_config.analytic_penalties = analytic_penalties;
    fit_config.topology_auto_selector = py_config
        .topology_auto_selector
        .as_ref()
        .map(gam::solver::topology_selector::TopologyAutoSelector::from_json)
        .transpose()?;
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
    if let Some(raw_gpu) = py_config.gpu {
        fit_config.gpu_policy = gam::gpu::GpuPolicy::parse(&raw_gpu).ok_or_else(|| {
            format!(
                "invalid gpu policy '{}'; supported values are auto, off, force",
                raw_gpu
            )
        })?;
    }
    if let Some(raw_device) = py_config.device {
        fit_config.device = gam::solver::gpu::Device::parse(&raw_device).ok_or_else(|| {
            format!(
                "invalid solver device '{}'; supported values are cpu, cuda",
                raw_device
            )
        })?;
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

fn parse_group_metadata(
    direct: Option<GroupMetadata>,
    groups: Option<serde_json::Value>,
) -> Result<Option<GroupMetadata>, String> {
    match (direct, groups) {
        (Some(metadata), None) => Ok(nonempty_group_metadata(metadata)),
        (None, Some(groups)) => group_metadata_from_groups(groups),
        (None, None) => Ok(None),
        (Some(_), Some(_)) => {
            Err("fit config accepts either group_metadata or groups metadata, not both".to_string())
        }
    }
}

fn parse_gamma_pair_value(
    label: &str,
    value: serde_json::Value,
) -> Result<(String, f64, f64), String> {
    match value {
        serde_json::Value::Array(values) => {
            if values.len() != 2 {
                return Err(format!(
                    "precision_hyperpriors['{label}'] must be [shape, rate]"
                ));
            }
            let shape = values[0]
                .as_f64()
                .ok_or_else(|| format!("precision_hyperpriors['{label}'][0] must be numeric"))?;
            let rate = values[1]
                .as_f64()
                .ok_or_else(|| format!("precision_hyperpriors['{label}'][1] must be numeric"))?;
            Ok((label.to_string(), shape, rate))
        }
        serde_json::Value::Object(mut map) => {
            let shape = map
                .remove("shape")
                .or_else(|| map.remove("a"))
                .or_else(|| map.remove("a_p"))
                .ok_or_else(|| format!("precision_hyperpriors['{label}'] missing shape/a"))?
                .as_f64()
                .ok_or_else(|| {
                    format!("precision_hyperpriors['{label}'] shape/a must be numeric")
                })?;
            let rate = map
                .remove("rate")
                .or_else(|| map.remove("b"))
                .or_else(|| map.remove("b_p"))
                .ok_or_else(|| format!("precision_hyperpriors['{label}'] missing rate/b"))?
                .as_f64()
                .ok_or_else(|| {
                    format!("precision_hyperpriors['{label}'] rate/b must be numeric")
                })?;
            Ok((label.to_string(), shape, rate))
        }
        _ => Err(format!(
            "precision_hyperpriors['{label}'] must be [shape, rate] or an object"
        )),
    }
}

fn parse_precision_hyperpriors(
    precision_hyperpriors: Option<serde_json::Value>,
    penalty_block_gamma_priors: Option<serde_json::Value>,
) -> Result<Vec<(String, f64, f64)>, String> {
    let raw = match (precision_hyperpriors, penalty_block_gamma_priors) {
        (Some(_), Some(_)) => {
            return Err(
                "fit config accepts either precision_hyperpriors or penalty_block_gamma_priors, not both"
                    .to_string(),
            );
        }
        (Some(raw), None) | (None, Some(raw)) => raw,
        (None, None) => {
            return Ok(Vec::new());
        }
    };
    let raw_name = "precision_hyperpriors";
    let Some(raw) = (match raw {
        serde_json::Value::Null => None,
        other => Some(other),
    }) else {
        return Ok(Vec::new());
    };
    match raw {
        serde_json::Value::Object(map) => map
            .into_iter()
            .map(|(label, value)| parse_gamma_pair_value(&label, value))
            .collect(),
        serde_json::Value::Array(items) => items
            .into_iter()
            .enumerate()
            .map(|(idx, item)| {
                match item {
                    serde_json::Value::Object(mut obj) => {
                        let label = obj
                            .remove("label")
                            .or_else(|| obj.remove("name"))
                            .or_else(|| obj.remove("group"))
                            .ok_or_else(|| {
                                format!("{raw_name}[{idx}] needs label/name/group")
                            })?;
                        let serde_json::Value::String(label) = label else {
                            return Err(format!("{raw_name}[{idx}] label must be a string"));
                        };
                        parse_gamma_pair_value(&label, serde_json::Value::Object(obj))
                    }
                    serde_json::Value::Array(mut values) => {
                        if values.len() != 2 && values.len() != 3 {
                            return Err(format!(
                                "{raw_name}[{idx}] must be [label, shape, rate] or [label, [shape, rate]]"
                            ));
                        }
                        let label = values.remove(0);
                        let serde_json::Value::String(label) = label else {
                            return Err(format!("{raw_name}[{idx}][0] must be a string label"));
                        };
                        let pair = if values.len() == 1 {
                            values.remove(0)
                        } else {
                            serde_json::Value::Array(values)
                        };
                        parse_gamma_pair_value(&label, pair)
                    }
                    _ => Err(format!("{raw_name}[{idx}] must be an object or array")),
                }
            })
            .collect(),
        _ => Err(format!("{raw_name} must be a map or array")),
    }
}

fn nonempty_group_metadata(metadata: GroupMetadata) -> Option<GroupMetadata> {
    if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    }
}

fn group_metadata_from_groups(groups: serde_json::Value) -> Result<Option<GroupMetadata>, String> {
    match groups {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Object(map) => {
            let out = map.into_iter().collect::<BTreeMap<_, _>>();
            Ok(nonempty_group_metadata(out))
        }
        serde_json::Value::Array(items) => {
            let mut out = BTreeMap::new();
            for (idx, item) in items.into_iter().enumerate() {
                let serde_json::Value::Object(mut group) = item else {
                    return Err(format!("groups[{idx}] must be an object"));
                };
                let Some(metadata) = group.remove("metadata") else {
                    continue;
                };
                let name = group
                    .remove("name")
                    .or_else(|| group.remove("id"))
                    .or_else(|| group.remove("key"))
                    .ok_or_else(|| {
                        format!(
                            "groups[{idx}] with metadata must include a string name, id, or key"
                        )
                    })?;
                let serde_json::Value::String(name) = name else {
                    return Err(format!("groups[{idx}] name/id/key must be a string"));
                };
                if name.is_empty() {
                    return Err(format!("groups[{idx}] name/id/key must be non-empty"));
                }
                if out.insert(name.clone(), metadata).is_some() {
                    return Err(format!("duplicate group metadata key '{name}'"));
                }
            }
            Ok(nonempty_group_metadata(out))
        }
        _ => Err("groups must be an object map or an array of group objects".to_string()),
    }
}

fn request_metadata(request: &FitRequest<'_>) -> (&'static str, &'static str, bool) {
    match request {
        FitRequest::Standard(standard_request) => {
            (standard_request.family.pretty_name(), "standard", true)
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
        FitRequest::SurvivalTransformation(request) => {
            // cause_count is Result<usize, SurvivalError>; on error fall through
            // to the non-competing-risks branches (display-only path).
            let cause_count =
                gam::survival::cause_count_from_event_codes(request.spec.event_target.view())
                    .unwrap_or(1);
            if cause_count > 1 {
                ("Cause-specific survival", "competing risks survival", true)
            } else {
                match request.spec.likelihood_mode {
                    gam::families::survival_construction::SurvivalLikelihoodMode::Weibull => {
                        ("Survival Weibull", "survival", true)
                    }
                    _ => ("Survival", "survival", true),
                }
            }
        }
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
    let records = string_records_from_rows(&headers, &rows)?;
    let records_ms = t_records.elapsed().as_secs_f64() * 1000.0;
    if records_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] ffi_string_records | n_rows={} | n_cols={} | {:.1}ms",
            n_rows,
            n_cols,
            records_ms
        );
    }
    drop(rows);
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
    headers: &[String],
    rows: &[Vec<String>],
) -> Result<EncodedDataset, String> {
    // Headers-only validation up-front (mirrors the missing-column path in
    // schema_check). We deliberately skip the schema_check call here because
    // schema_check would otherwise do a full encode internally, doubling the
    // O(N·p) FFI ingest cost. Any encoding errors below surface directly.
    let expected_names = required_prediction_columns(model)?;
    let present_names = headers.iter().cloned().collect::<BTreeSet<_>>();
    let missing = expected_names
        .difference(&present_names)
        .map(|name| format!("missing required column '{name}'"))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(missing.join(" "));
    }
    let schema = model.require_data_schema()?;
    let records = string_records_from_rows(headers, rows)?;
    encode_recordswith_schema(
        headers.to_vec(),
        records,
        schema,
        UnseenCategoryPolicy::Error,
    )
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
        let records = string_records_from_rows(headers, rows)?;
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
            _ if payload
                .survival_cause_count
                .is_some_and(|cause_count| cause_count > 1) =>
            {
                "competing risks survival".to_string()
            }
            Some("marginal-slope") => "survival marginal-slope".to_string(),
            Some("location-scale") => "survival location-scale".to_string(),
            None
            | Some("latent")
            | Some("latent-binary")
            | Some("transformation")
            | Some("weibull")
            | Some("royston-parmar")
            | Some(_) => model.predict_model_class().name().to_string(),
        },
        FittedFamily::LatentSurvival { .. } => "latent survival".to_string(),
        FittedFamily::LatentBinary { .. } => "latent binary".to_string(),
        FittedFamily::Standard { .. }
        | FittedFamily::LocationScale { .. }
        | FittedFamily::MarginalSlope { .. }
        | FittedFamily::TransformationNormal { .. } => {
            model.predict_model_class().name().to_string()
        }
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
            ParsedTerm::LogSlopeSurface { z_column, terms } => {
                required.insert(z_column.clone());
                add_formula_term_columns(required, terms);
            }
        }
    }
}

fn string_records_from_rows(
    headers: &[String],
    rows: &[Vec<String>],
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
    rows.iter()
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
            Ok(StringRecord::from(row.clone()))
        })
        .collect()
}

fn periodic_bspline_basis_dense_via_spec(
    t: ArrayView1<'_, f64>,
    domain: (f64, f64),
    degree: usize,
    num_basis: usize,
) -> Result<Array2<f64>, String> {
    let (left, right) = domain;
    let period = right - left;
    if !(period.is_finite() && period > 0.0) {
        return Err(format!(
            "periodic B-spline domain must be a finite ordered interval; got ({left}, {right})"
        ));
    }
    let spec = PeriodicBSplineBasisSpec::new(degree, num_basis, period, left, 0);
    build_periodic_bspline_basis_1d(t, &spec)
        .map_err(|err| format!("failed to evaluate periodic B-spline basis: {err}"))
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
        periodic_bspline_basis_dense_via_spec(t, (left, right), degree, num_basis)
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
        return Err(
            "periodic B-spline first-derivative as a dense (N, K) matrix is no longer exposed; \
             use periodic_bspline_input_location_first_derivative for the (N, K, 1) jet"
                .to_string(),
        );
    }
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
        periodic: None,
        length_scale: None,
        power: 0.0,
        nullspace_order: duchon_nullspace_from_m(m),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    if periodic {
        let built = build_duchon_basis_mixed_periodicity_auto(data.view(), &spec, &[true], None)
            .map_err(|err| format!("failed to evaluate Duchon basis: {err}"))?;
        return built
            .design
            .try_to_dense_by_chunks("duchon_basis_1d_impl")
            .map_err(|err| format!("failed to evaluate Duchon basis: {err}"));
    }
    let built = build_duchon_basis(data.view(), &spec)
        .map_err(|err| format!("failed to evaluate Duchon basis: {err}"))?;
    built
        .design
        .try_to_dense_by_chunks("duchon_basis_1d_impl")
        .map_err(|err| format!("failed to evaluate Duchon basis: {err}"))
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
        0.0,
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
            None,
            None,
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
                0.0,
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
            0.08, 0.16, 0.27, 0.39, 0.51, 0.64, 0.76, 0.89, 0.10, 0.19, 0.31, 0.43, 0.56, 0.68,
            0.80, 0.92
        ];
        let y = Array2::from_shape_fn((t.len(), 2), |(row, output)| {
            let u = t[row];
            let scale = output as f64 + 1.0;
            0.3 + 0.4 * scale * u + 0.15 * (2.0 * u + 0.2 * scale).sin()
        });
        let offsets = array![0_usize, 8_usize, 16_usize];
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let penalty = Array2::from_diag(&array![0.2, 0.35, 0.8, 1.4, 2.2, 3.1, 4.0]);
        let weights = array![
            1.0, 0.9, 1.1, 1.2, 0.85, 1.05, 0.95, 1.07, 1.0, 1.15, 0.88, 1.04, 0.93, 1.08, 0.97,
            1.12
        ];
        let grad_lambda = array![0.0, 0.0];
        let grad_reml_score = array![0.0, 0.0];
        let grad_coefficients = Array3::zeros((2, penalty.nrows(), y.ncols()));
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
            None,
            0,
            None,
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
    Ok::<(), _>(())
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

fn pyffi_duchon_previous_nullspace_order(order: DuchonNullspaceOrder) -> DuchonNullspaceOrder {
    match order {
        DuchonNullspaceOrder::Zero => DuchonNullspaceOrder::Zero,
        DuchonNullspaceOrder::Linear => DuchonNullspaceOrder::Zero,
        DuchonNullspaceOrder::Degree(2) => DuchonNullspaceOrder::Linear,
        DuchonNullspaceOrder::Degree(k) => DuchonNullspaceOrder::Degree(k - 1),
    }
}

fn pyffi_duchon_polynomial_block(
    points: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Array2<f64> {
    let n_rows = points.nrows();
    let dim = points.ncols();
    match order {
        DuchonNullspaceOrder::Zero => Array2::<f64>::ones((n_rows, 1)),
        DuchonNullspaceOrder::Linear => {
            let mut poly = Array2::<f64>::zeros((n_rows, dim + 1));
            poly.column_mut(0).fill(1.0);
            for axis in 0..dim {
                poly.column_mut(axis + 1).assign(&points.column(axis));
            }
            poly
        }
        DuchonNullspaceOrder::Degree(degree) => {
            let exponents = pyffi_duchon_monomial_exponents(dim, degree);
            let mut poly = Array2::<f64>::zeros((n_rows, exponents.len()));
            for (col, alpha) in exponents.iter().enumerate() {
                for row in 0..n_rows {
                    let mut value = 1.0;
                    for axis in 0..dim {
                        let exp = alpha[axis];
                        if exp != 0 {
                            value *= points[[row, axis]].powi(exp as i32);
                        }
                    }
                    poly[[row, col]] = value;
                }
            }
            poly
        }
    }
}

fn pyffi_duchon_monomial_exponents(
    dimension: usize,
    max_total_degree: usize,
) -> Vec<Vec<usize>> {
    fn recurse(
        axis: usize,
        remaining_degree: usize,
        current: &mut [usize],
        out: &mut Vec<Vec<usize>>,
    ) {
        if axis + 1 == current.len() {
            current[axis] = remaining_degree;
            out.push(current.to_vec());
            return;
        }
        for exponent in (0..=remaining_degree).rev() {
            current[axis] = exponent;
            recurse(axis + 1, remaining_degree - exponent, current, out);
        }
    }

    if dimension == 0 {
        return vec![Vec::new()];
    }

    let mut out = Vec::new();
    let mut current = vec![0usize; dimension];
    for total_degree in 0..=max_total_degree {
        recurse(0, total_degree, &mut current, &mut out);
    }
    out
}

fn pyffi_duchon_effective_nullspace_order(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> DuchonNullspaceOrder {
    let mut effective = order;
    while effective != DuchonNullspaceOrder::Zero
        && centers.nrows() <= pyffi_duchon_polynomial_block(centers, effective).ncols()
    {
        effective = pyffi_duchon_previous_nullspace_order(effective);
    }
    effective
}

fn pyffi_duchon_kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Result<Array2<f64>, String> {
    let polynomial_block = pyffi_duchon_polynomial_block(centers, order);
    gam::faer_ndarray::rrqr_nullspace_basis(
        &polynomial_block,
        gam::faer_ndarray::default_rrqr_rank_alpha(),
    )
    .map(|(null_basis, _)| null_basis)
    .map_err(|err| format!("failed to build Duchon kernel constraint nullspace: {err}"))
}

/// Parse the optional ``nullspace_order`` keyword on the primitive Duchon
/// bindings. ``None`` falls back to the legacy default derived from ``m``
/// (so existing call sites stay bit-identical). Accepted strings:
/// ``"zero"`` (constant nullspace), ``"linear"`` (constant + linear),
/// or ``"degree<k>"`` for k ≥ 2 (polynomials of total degree ≤ k).
fn parse_nullspace_order(raw: Option<&str>, m: usize) -> PyResult<DuchonNullspaceOrder> {
    let Some(raw) = raw else {
        return Ok(duchon_nullspace_from_m(m));
    };
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "zero" => Ok(DuchonNullspaceOrder::Zero),
        "linear" => Ok(DuchonNullspaceOrder::Linear),
        other if other.starts_with("degree") => {
            let suffix = other.trim_start_matches("degree").trim_start_matches('=');
            let k: usize = suffix.parse().map_err(|_| {
                py_value_error(format!(
                    "invalid nullspace_order '{raw}'; expected 'zero', 'linear', or 'degree<k>' with integer k ≥ 2"
                ))
            })?;
            if k < 2 {
                return Err(py_value_error(format!(
                    "nullspace_order 'degree{k}' must have k ≥ 2; use 'zero' or 'linear' instead"
                )));
            }
            Ok(DuchonNullspaceOrder::Degree(k))
        }
        _ => Err(py_value_error(format!(
            "invalid nullspace_order '{raw}'; expected 'zero', 'linear', or 'degree<k>'"
        ))),
    }
}

/// Resolved Duchon hybrid-mode configuration as derived from the public
/// keyword surface (``length_scale``, ``nullspace_order``, ``power``).
/// All three primitives share this resolver so the auto-resolution policy
/// matches the formula path 1:1.
struct DuchonHybridConfig {
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: f64,
}

/// Resolve the (nullspace_order, power) pair given the optional public
/// keywords. Auto-resolution uses :func:`resolve_duchon_orders` with
/// ``max_operator_derivative_order = 2`` to match the formula API's
/// default: the non-periodic Duchon basis builder downstream of these
/// primitives instantiates the full operator-penalty triplet
/// (mass + tension + stiffness), and stiffness collocation requires
/// ``2(p + s) > d + 2``. Using max_op=2 here keeps the auto-resolved
/// power large enough that the kernel validator inside
/// ``build_duchon_basis`` accepts the spec.
fn resolve_duchon_hybrid_config(
    dim: usize,
    m: usize,
    length_scale: Option<f64>,
    nullspace_order: Option<&str>,
    explicit_power: Option<f64>,
) -> PyResult<DuchonHybridConfig> {
    let requested_nullspace = parse_nullspace_order(nullspace_order, m)?;
    let (resolved_nullspace, auto_power) =
        resolve_duchon_orders(dim, requested_nullspace, 2, length_scale);
    let power = explicit_power.unwrap_or(auto_power as f64);
    Ok(DuchonHybridConfig {
        length_scale,
        nullspace_order: resolved_nullspace,
        power,
    })
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

fn fit_with_null_space_logdet(
    design: &TermCollectionDesign,
    fit: &gam::estimate::UnifiedFitResult,
) -> Result<gam::estimate::UnifiedFitResult, String> {
    let mut fit = fit.clone();
    let (null_dim, logdet) = compute_null_space_metadata(design, &fit)?;
    fit.artifacts.null_space_dim = Some(null_dim);
    fit.artifacts.null_space_logdet = Some(logdet);
    Ok(fit)
}

fn compute_null_space_metadata(
    design: &TermCollectionDesign,
    fit: &gam::estimate::UnifiedFitResult,
) -> Result<(usize, f64), String> {
    let hessian = fit
        .penalized_hessian()
        .ok_or_else(|| "null-space Hessian logdet requires fitted penalized Hessian".to_string())?;
    let p = hessian.nrows();
    if hessian.ncols() != p {
        return Err(format!(
            "null-space Hessian logdet requires a square Hessian, got {}x{}",
            hessian.nrows(),
            hessian.ncols()
        ));
    }
    if p != design.design.ncols() {
        return Err(format!(
            "null-space Hessian logdet design/Hessian mismatch: design has {} columns, Hessian is {}x{}",
            design.design.ncols(),
            hessian.nrows(),
            hessian.ncols()
        ));
    }

    let mut penalty = Array2::<f64>::zeros((p, p));
    for (idx, block) in design.penalties.iter().enumerate() {
        let range = block.col_range.clone();
        if range.end > p || block.local.nrows() != range.len() || block.local.ncols() != range.len()
        {
            return Err(format!(
                "null-space Hessian logdet penalty {idx} shape mismatch: range {}..{}, local {}x{}, p={p}",
                range.start,
                range.end,
                block.local.nrows(),
                block.local.ncols()
            ));
        }
        penalty
            .slice_mut(s![range.clone(), range])
            .scaled_add(1.0, &block.local);
    }

    let (null_basis, _) = gam::faer_ndarray::rrqr_nullspace_basis(
        &penalty,
        gam::faer_ndarray::default_rrqr_rank_alpha(),
    )
    .map_err(|err| format!("failed to compute penalty null-space basis: {err}"))?;
    let q = null_basis.ncols();
    if q == 0 {
        return Ok((0, 0.0));
    }

    let projected = hessian.dot(&null_basis);
    let mut restricted = null_basis.t().dot(&projected);
    restricted = (&restricted + &restricted.t()) * 0.5;
    let chol = restricted
        .cholesky(Side::Lower)
        .map_err(|err| format!("null-space Hessian is not positive definite: {err}"))?;
    let logdet = 2.0 * chol.diag().iter().map(|value| value.ln()).sum::<f64>();
    if logdet.is_finite() {
        Ok((q, logdet))
    } else {
        Err(format!("null-space Hessian logdet is not finite: {logdet}"))
    }
}

fn build_standard_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    family: LikelihoodSpec,
    saved_fit: &gam::estimate::UnifiedFitResult,
    design: &TermCollectionDesign,
    resolved_termspec: TermCollectionSpec,
    adaptive_regularization_diagnostics: Option<gam::smooth::AdaptiveRegularizationDiagnostics>,
    wiggle_knots: Option<Vec<f64>>,
    wiggle_degree: Option<usize>,
) -> Result<FittedModelPayload, String> {
    let saved_fit = fit_with_null_space_logdet(design, saved_fit)?;
    let latent_cloglog_state = if matches!(
        (&family.response, &family.link),
        (ResponseFamily::Binomial, InverseLink::LatentCLogLog(_))
    ) {
        Some(saved_latent_cloglog_state_from_fit(&saved_fit).expect(
            "latent-cloglog-binomial fit must produce an explicit latent-cloglog state",
        ))
    } else {
        saved_latent_cloglog_state_from_fit(&saved_fit)
    };
    let family_link = family.link_function();
    let family_name = family.name().to_string();
    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: family,
            link: Some(family_link),
            latent_cloglog_state,
            mixture_state: saved_mixture_state_from_fit(&saved_fit),
            sas_state: saved_sas_state_from_fit(&saved_fit),
        },
        family_name,
    );
    payload.unified = Some(saved_fit.clone());
    payload.fit_result = Some(saved_fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some(InverseLink::Standard(family_link));
    payload.linkwiggle_knots = wiggle_knots;
    payload.linkwiggle_degree = wiggle_degree;
    payload.set_training_feature_metadata(dataset.headers.clone(), dataset.feature_ranges());
    payload.resolved_termspec = Some(resolved_termspec);
    payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    Ok(payload)
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
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(LinkFunction::Identity),
            ),
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

    let likelihood = inverse_link_to_binomial_spec(&base_link).map_err(|e| {
        format!(
            "failed to resolve LikelihoodSpec for bernoulli marginal-slope base link {:?}: {e}",
            base_link
        )
    })?;

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
    payload.link = Some(base_link.clone());
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
    use gam::families::survival_construction::{
        build_survival_time_basis, parse_survival_baseline_config, parse_survival_likelihood_mode,
        parse_survival_time_basis_config, resolve_survival_time_anchor_value,
        survival_baseline_targetname,
    };

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
    let time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some((
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )),
    )?;

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(LinkFunction::Identity),
            ),
            survival_likelihood: Some("marginal-slope".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
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
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(baseline_cfg.target).to_string());
    payload.survival_baseline_scale = baseline_cfg.scale;
    payload.survival_baseline_shape = baseline_cfg.shape;
    payload.survival_baseline_rate = baseline_cfg.rate;
    payload.survival_baseline_makeham = baseline_cfg.makeham;
    payload.apply_survival_time_basis(&SavedSurvivalTimeBasis::from_build(
        &time_build,
        time_anchor,
    ));
    payload.survivalridge_lambda = Some(fit_config.ridge_lambda);
    payload.survival_likelihood = Some(survival_likelihood_modename(likelihood_mode).to_string());
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    payload.link = Some(InverseLink::Standard(LinkFunction::Probit));
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
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(LinkFunction::Identity),
            ),
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
    let cause_count = rp_result.fit.blocks.len().max(1);
    let is_joint_cause_specific = cause_count > 1;
    payload.survivalspec = Some(if is_joint_cause_specific {
        "cause-specific".to_string()
    } else {
        "net".to_string()
    });
    if is_joint_cause_specific {
        payload.survival_cause_count = Some(cause_count);
        payload.survival_endpoint_names = Some(
            (1..=cause_count)
                .map(|idx| format!("cause_{idx}"))
                .collect(),
        );
    }
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(rp_result.baseline_cfg.target).to_string());
    payload.survival_baseline_scale = rp_result.baseline_cfg.scale;
    payload.survival_baseline_shape = rp_result.baseline_cfg.shape;
    payload.survival_baseline_rate = rp_result.baseline_cfg.rate;
    payload.survival_baseline_makeham = rp_result.baseline_cfg.makeham;
    payload.apply_survival_time_basis(&rp_result.time_basis);
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
        let start = rp_result.time_base_ncols;
        let end = start + timewiggle.ncols;
        if is_joint_cause_specific {
            let mut by_cause = Vec::with_capacity(cause_count);
            for (cause_idx, block) in rp_result.fit.blocks.iter().enumerate() {
                if block.beta.len() < end {
                    return Err(format!(
                        "joint cause-specific survival timewiggle beta mismatch for cause {}: beta has {}, needs {end}",
                        cause_idx + 1,
                        block.beta.len()
                    ));
                }
                by_cause.push(block.beta.slice(s![start..end]).to_vec());
            }
            payload.beta_baseline_timewiggle_by_cause = Some(by_cause);
        } else {
            let beta = &rp_result.fit.beta;
            if beta.len() < end {
                return Err(format!(
                    "survival transformation timewiggle beta mismatch: beta has {}, needs {end}",
                    beta.len()
                ));
            }
            payload.beta_baseline_timewiggle = Some(beta.slice(s![start..end]).to_vec());
        }
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
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(LinkFunction::Identity),
            ),
            base_link: None,
        },
        "gaussian-location-scale".to_string(),
    );
    payload.unified = Some(fit.clone());
    payload.fit_result = Some(fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some(InverseLink::Standard(LinkFunction::Identity));
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

    let dense_mean = ls_result
        .fit
        .mean_design
        .design
        .try_to_dense_by_chunks("binomial location-scale mean design")?;
    let dense_noise = ls_result
        .fit
        .noise_design
        .design
        .try_to_dense_by_chunks("binomial location-scale noise design")?;
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

    let location_scale_likelihood = inverse_link_to_binomial_spec(&link_kind).map_err(|e| {
        format!(
            "failed to resolve LikelihoodSpec for binomial location-scale link {:?}: {e}",
            link_kind
        )
    })?;
    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: location_scale_likelihood,
            base_link: Some(link_kind.clone()),
        },
        "binomial-location-scale".to_string(),
    );
    payload.unified = Some(fit.clone());
    payload.fit_result = Some(fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some(link_kind.clone());
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
    let dense_threshold = ls_result
        .fit
        .threshold_design
        .design
        .try_to_dense_by_chunks("survival location-scale threshold design")?;
    let dense_log_sigma = ls_result
        .fit
        .log_sigma_design
        .design
        .try_to_dense_by_chunks("survival location-scale log_sigma design")?;
    let dense_time_exit = time_build
        .x_exit_time
        .try_to_dense_by_chunks("survival location-scale time-exit design")?;
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
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(LinkFunction::Identity),
            ),
            survival_likelihood: Some(survival_likelihood_modename(likelihood_mode).to_string()),
            survival_distribution: residual_distribution_from_inverse_link(&fitted_inverse_link),
            frailty: gam::families::lognormal_kernel::FrailtySpec::None,
        },
        "royston-parmar".to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some(fitted_inverse_link.clone());
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
    payload.apply_survival_time_basis(&SavedSurvivalTimeBasis::from_build(
        &time_build,
        time_anchor,
    ));
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
    payload.survival_distribution = residual_distribution_from_inverse_link(&fitted_inverse_link);
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
    payload.apply_survival_time_basis(&SavedSurvivalTimeBasis::from_build(
        &time_build,
        time_anchor,
    ));
    payload.survival_likelihood = Some(likelihood_label);
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
    if model
        .payload()
        .survival_cause_count
        .is_some_and(|cause_count| cause_count > 1)
    {
        let result = predict_competing_risks_survival_result(model, dataset, options)?;
        return serialize_competing_risks_prediction_payload(result);
    }
    let result = predict_survival_result(model, dataset, options)?;
    serialize_survival_prediction_payload(model, result)
}

fn predict_competing_risks_survival_result(
    model: &FittedModel,
    dataset: &EncodedDataset,
    options: &PyPredictOptions,
) -> Result<gam::survival_predict::CompetingRisksPredictResult, String> {
    use gam::survival_predict::{SurvivalPredictRequest, predict_competing_risks_survival};

    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let primary_offset =
        resolve_offset_column(dataset, &col_map, payload.offset_column.as_deref())?;
    let noise_offset = ndarray::Array1::<f64>::zeros(dataset.values.nrows());
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
    Ok(predict_competing_risks_survival(request)?)
}

fn predict_survival_result(
    model: &FittedModel,
    dataset: &EncodedDataset,
    options: &PyPredictOptions,
) -> Result<gam::survival_predict::SurvivalPredictResult, String> {
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
    Ok(predict_survival(request)?)
}

fn serialize_survival_prediction_payload(
    model: &FittedModel,
    result: gam::survival_predict::SurvivalPredictResult,
) -> Result<String, String> {
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
        gam::survival_construction::SurvivalLikelihoodMode::Transformation
        | gam::survival_construction::SurvivalLikelihoodMode::Weibull
        | gam::survival_construction::SurvivalLikelihoodMode::LatentBinary => {
            model.predict_model_class().name().to_string()
        }
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

fn serialize_competing_risks_prediction_payload(
    result: gam::survival_predict::CompetingRisksPredictResult,
) -> Result<String, String> {
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    for (endpoint_idx, name) in result.endpoint_names.iter().enumerate() {
        let suffix = name.replace('-', "_");
        columns.insert(
            format!("eta_{suffix}"),
            result.linear_predictor[endpoint_idx].to_vec(),
        );
        let t_last = result.cif[endpoint_idx].ncols().saturating_sub(1);
        columns.insert(
            format!("failure_prob_{suffix}"),
            (0..result.cif[endpoint_idx].nrows())
                .map(|i| result.cif[endpoint_idx][[i, t_last]])
                .collect(),
        );
    }
    let t_last = result.overall_survival.ncols().saturating_sub(1);
    columns.insert(
        "overall_survival".to_string(),
        (0..result.overall_survival.nrows())
            .map(|i| result.overall_survival[[i, t_last]])
            .collect(),
    );
    let likelihood_mode_str = match result.likelihood_mode {
        gam::survival_construction::SurvivalLikelihoodMode::MarginalSlope => "marginal-slope",
        gam::survival_construction::SurvivalLikelihoodMode::LocationScale => "location-scale",
        gam::survival_construction::SurvivalLikelihoodMode::Transformation => "transformation",
        gam::survival_construction::SurvivalLikelihoodMode::Weibull => "weibull",
        gam::survival_construction::SurvivalLikelihoodMode::Latent => "latent",
        gam::survival_construction::SurvivalLikelihoodMode::LatentBinary => "latent-binary",
    };
    let payload = serde_json::json!({
        "class": "competing_risks_prediction",
        "model_class": "competing risks survival",
        "likelihood_mode": likelihood_mode_str,
        "endpoint_names": result.endpoint_names,
        "times": result.times,
        "hazard": matrices_to_nested(&result.hazard),
        "survival": matrices_to_nested(&result.survival),
        "cumulative_hazard": matrices_to_nested(&result.cumulative_hazard),
        "cif": matrices_to_nested(&result.cif),
        "overall_survival": matrix_to_nested(&result.overall_survival),
        "linear_predictor": result
            .linear_predictor
            .iter()
            .map(|eta| eta.to_vec())
            .collect::<Vec<_>>(),
        "columns": columns,
    });
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize competing-risks prediction payload: {err}"))
}

fn matrix_to_nested(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|i| (0..matrix.ncols()).map(|j| matrix[[i, j]]).collect())
        .collect()
}

fn matrices_to_nested(matrices: &[Array2<f64>]) -> Vec<Vec<Vec<f64>>> {
    matrices.iter().map(matrix_to_nested).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn symmetric_curvature_solve_preserves_negative_modes() {
        let matrix = array![[2.0, 0.0, 0.0], [0.0, -4.0, 0.0], [0.0, 0.0, -1.0e-15]];
        let rhs = array![8.0, -8.0, 1.0];
        let solved = solve_symmetric_vector_with_floor(&matrix, &rhs, 1.0e-3)
            .expect("indefinite symmetric curvature solve");

        assert!((solved[0] - 4.0).abs() <= 1.0e-12);
        assert!((solved[1] - 2.0).abs() <= 1.0e-12);
        assert!((solved[2] + 250.0).abs() <= 1.0e-9);
    }

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
                likelihood: LikelihoodSpec::new(
                    ResponseFamily::Gaussian,
                    InverseLink::Standard(LinkFunction::Identity),
                ),
                link: Some(LinkFunction::Identity),
                latent_cloglog_state: None,
                mixture_state: None,
                sas_state: None,
            },
            "gaussian".to_string(),
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
        // The truth must NOT lie (essentially) in span(X). The design here is
        // {1, t, t² - 0.45, sin(0.9t) + 0.05t, cos(1.2t) - 0.15t²}; a smooth
        // polynomial plus low-frequency cos would be fit nearly to machine
        // precision, driving σ² → 0 and ∂(score)/∂y ≈ ν w r / dp → ∞, at which
        // point central FD with Richardson extrapolation cannot resolve the
        // (analytic, exact) gradient at 1e-6 relative because the truncation
        // term scales with f⁽⁵⁾(y). The high-frequency sin term below lies
        // outside that span on t ∈ [-1.19, 1.31] and leaves a genuine residual
        // so the analytic-vs-FD comparison is meaningful at strict tolerance.
        Array2::from_shape_fn((20, 3), |(row, output)| {
            let t = (row as f64 - 9.5) / 8.0;
            let phase = output as f64 + 1.0;
            0.2 + 0.35 * phase * t
                + 0.18 * t * t
                + (0.4 + 0.1 * phase) * (0.8 * t + 0.25 * phase).cos()
                + 0.07 * (6.5 * t + 0.4 * phase).sin()
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
            1.0 + 0.07 * (1.3 * t).cos() + 0.025 * t
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
        init_lambda: Option<f64>,
    ) -> f64 {
        let gated_x = apply_by_gate(x, by, 1).expect("by-gated design");
        let fit = gaussian_reml_multi_closed_form_with_cache(
            gated_x.view(),
            y,
            penalty,
            Some(weights),
            init_lambda,
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
            0.0,
        )
        .expect("by-gated analytic backward");
        let (grad_x, grad_by) =
            apply_by_gate_backward(x, by, 1, backward.grad_x.view()).expect("by-gate backward");
        (grad_x, backward.grad_y, grad_by, backward.grad_weights)
    }

    fn assert_fd_estimate_close(
        label: &str,
        analytic: f64,
        finite_difference: f64,
        finite_difference_error: f64,
    ) {
        let rel_tol = 1.0e-6_f64;
        let abs_tol = 1.0e-6_f64;
        // Loosen the tolerance by the caller's own FD-error estimate so the
        // assertion does not flag differences that the FD discretization
        // itself cannot resolve. The base abs/rel envelope handles
        // analytic-side rounding; `finite_difference_error` covers
        // truncation error from the FD stencil.
        let tol = abs_tol
            .max(rel_tol * analytic.abs().max(finite_difference.abs()))
            .max(finite_difference_error.abs());
        let diff = (analytic - finite_difference).abs();
        assert!(
            diff <= tol,
            "{label}: analytic={analytic:.12e}, finite_difference={finite_difference:.12e}, diff={diff:.3e}, tol={tol:.3e}"
        );
    }

    fn adaptive_finite_difference<F>(center: f64, step: f64, mut objective: F) -> (f64, f64)
    where
        F: FnMut(f64) -> f64,
    {
        let multipliers = [100.0_f64, 50.0, 25.0, 12.5, 6.25];
        let scale = step * center.abs().max(1.0);
        let mut best = f64::NAN;
        let mut best_delta = f64::INFINITY;
        let mut previous: Option<f64> = None;
        for multiplier in multipliers {
            let h = multiplier * scale;
            let coarse = (objective(center + h) - objective(center - h)) / (2.0 * h);
            let half_h = 0.5 * h;
            let fine = (objective(center + half_h) - objective(center - half_h)) / (2.0 * half_h);
            let estimate = fine + (fine - coarse) / 3.0;
            if let Some(prev) = previous {
                let delta = (estimate - prev).abs();
                if delta < best_delta {
                    best_delta = delta;
                    best = estimate;
                }
            } else {
                best = estimate;
            }
            previous = Some(estimate);
        }
        (best, best_delta)
    }

    fn blocks_reml_sign_inputs() -> (Vec<Array2<f64>>, Vec<Array2<f64>>, Array1<f64>, Array1<f64>) {
        let n = 18;
        let x1 = Array2::from_shape_fn((n, 3), |(row, col)| {
            let t = (row as f64 + 0.5) / n as f64;
            match col {
                0 => 1.0,
                1 => t - 0.45,
                2 => (4.3 * t).sin() + 0.2 * t,
                _ => unreachable!(),
            }
        });
        let x2 = Array2::from_shape_fn((n, 2), |(row, col)| {
            let t = (row as f64 + 0.5) / n as f64;
            match col {
                0 => (2.1 * t).cos() - 0.1,
                1 => (7.7 * t + 0.3).sin(),
                _ => unreachable!(),
            }
        });
        let s1 = array![[0.7, 0.08, 0.02], [0.08, 1.3, -0.04], [0.02, -0.04, 2.1],];
        let s2 = array![[0.9, -0.06], [-0.06, 1.8]];
        let y = Array1::from_shape_fn(n, |row| {
            let t = (row as f64 + 0.5) / n as f64;
            0.25 + 0.35 * t + 0.18 * (5.0 * t).sin() + 0.09 * (11.0 * t + 0.2).cos()
        });
        let weights = Array1::from_shape_fn(n, |row| {
            let t = (row as f64 + 0.5) / n as f64;
            0.9 + 0.16 * (1.7 * t).cos()
        });
        (vec![x1, x2], vec![s1, s2], y, weights)
    }

    fn gaussian_reml_fit_blocks_forward_native(
        designs: &[Array2<f64>],
        penalties: &[Array2<f64>],
        y: ArrayView1<'_, f64>,
        weights: ArrayView1<'_, f64>,
        init_rhos: &[f64],
    ) -> Result<
        (
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            f64,
            Array1<f64>,
        ),
        EstimationError,
    > {
        let n_rows = designs[0].nrows();
        let mut col_offsets = vec![0usize];
        for design in designs {
            col_offsets.push(col_offsets.last().copied().unwrap() + design.ncols());
        }
        let p_total = *col_offsets.last().unwrap();
        let mut joint_x = Array2::<f64>::zeros((n_rows, p_total));
        for (block, design) in designs.iter().enumerate() {
            joint_x
                .slice_mut(s![.., col_offsets[block]..col_offsets[block + 1]])
                .assign(design);
        }
        let s_list = penalties
            .iter()
            .enumerate()
            .map(|(block, penalty)| {
                gam::smooth::BlockwisePenalty::new(
                    col_offsets[block]..col_offsets[block + 1],
                    penalty.clone(),
                )
            })
            .collect::<Vec<_>>();
        let heuristic_lambdas = init_rhos.iter().map(|rho| rho.exp()).collect::<Vec<_>>();
        let opts = gam::estimate::FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 200,
            tol: 1.0e-9,
            nullspace_dims: vec![0; s_list.len()],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        };
        let offset = Array1::<f64>::zeros(n_rows);
        let fit = gam::estimate::fit_gamwith_heuristic_lambdas(
            joint_x.clone(),
            y,
            weights,
            offset.view(),
            &s_list,
            Some(heuristic_lambdas.as_slice()),
            LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(LinkFunction::Identity),
            ),
            &opts,
        )?;
        let beta = fit.beta.clone();
        let fitted = joint_x.dot(&beta);
        let lambdas = fit.lambdas.clone();
        let log_lambdas = lambdas.mapv(|lambda| lambda.max(1.0e-300).ln());
        let edf_vec = fit
            .inference
            .as_ref()
            .map(|inference| inference.edf_by_block.clone())
            .unwrap_or_else(|| vec![0.0; lambdas.len()]);
        let edf = if edf_vec.len() == lambdas.len() {
            Array1::from_vec(edf_vec)
        } else {
            Array1::zeros(lambdas.len())
        };
        Ok((beta, fitted, lambdas, log_lambdas, fit.reml_score, edf))
    }

    fn blocks_profile_reml_score(
        designs: &[Array2<f64>],
        penalties: &[Array2<f64>],
        y: ArrayView1<'_, f64>,
        weights: ArrayView1<'_, f64>,
        init_rhos: &[f64],
    ) -> f64 {
        let (_, _, _, _, reml_score, _) =
            gaussian_reml_fit_blocks_forward_native(designs, penalties, y, weights, init_rhos)
                .expect("multi-block Gaussian REML profile score");
        reml_score
    }

    #[test]
    fn blocks_negative_reml_score_backward_sign_matches_profile_perturbations() {
        assert!(file!().ends_with(".rs"));
        let (designs, penalties, y, weights) = blocks_reml_sign_inputs();
        let init_rhos = vec![0.2, -0.4];
        let (_, _, _, log_lambdas, _, _) = gaussian_reml_fit_blocks_forward_native(
            &designs,
            &penalties,
            y.view(),
            weights.view(),
            init_rhos.as_slice(),
        )
        .expect("base multi-block Gaussian REML fit");
        let rhos = log_lambdas.to_vec();
        let backward = gaussian_reml_fit_blocks_backward_analytic(
            &designs,
            &penalties,
            y.view(),
            weights.view(),
            rhos.as_slice(),
            None,
            None,
            None,
            None,
            1.0,
            None,
        )
        .expect("negative REML score analytic VJP");
        let eps = 1.0e-5;

        for row in [2_usize, 11] {
            let (fd, fd_error) = adaptive_finite_difference(y[row], eps, |candidate| {
                let mut yp = y.clone();
                yp[row] = candidate;
                blocks_profile_reml_score(
                    &designs,
                    &penalties,
                    yp.view(),
                    weights.view(),
                    rhos.as_slice(),
                )
            });
            assert_fd_estimate_close(
                &format!("negative REML y[{row}] sign"),
                backward.grad_y[[row, 0]],
                fd,
                fd_error,
            );
        }

        for (block, row, col) in [(0_usize, 5_usize, 2_usize), (1, 12, 1)] {
            let center = designs[block][[row, col]];
            let (fd, fd_error) = adaptive_finite_difference(center, eps, |candidate| {
                let mut xp = designs.clone();
                xp[block][[row, col]] = candidate;
                blocks_profile_reml_score(
                    &xp,
                    &penalties,
                    y.view(),
                    weights.view(),
                    rhos.as_slice(),
                )
            });
            assert_fd_estimate_close(
                &format!("negative REML X{block}[{row},{col}] sign"),
                backward.grad_designs[block][[row, col]],
                fd,
                fd_error,
            );
        }

        for (block, row, col) in [(0_usize, 1_usize, 1_usize), (1, 0, 1)] {
            let center = penalties[block][[row, col]];
            let (fd, fd_error) = adaptive_finite_difference(center, eps, |candidate| {
                let mut sp = penalties.clone();
                sp[block][[row, col]] = candidate;
                sp[block][[col, row]] = candidate;
                blocks_profile_reml_score(&designs, &sp, y.view(), weights.view(), rhos.as_slice())
            });
            let analytic = if row == col {
                backward.grad_penalties[block][[row, col]]
            } else {
                backward.grad_penalties[block][[row, col]]
                    + backward.grad_penalties[block][[col, row]]
            };
            assert_fd_estimate_close(
                &format!("negative REML S{block}[{row},{col}] sign"),
                analytic,
                fd,
                fd_error,
            );
        }

        for row in [3_usize, 14] {
            let (fd, fd_error) = adaptive_finite_difference(weights[row], eps, |candidate| {
                let mut wp = weights.clone();
                wp[row] = candidate;
                blocks_profile_reml_score(
                    &designs,
                    &penalties,
                    y.view(),
                    wp.view(),
                    rhos.as_slice(),
                )
            });
            assert_fd_estimate_close(
                &format!("negative REML weight[{row}] sign"),
                backward.grad_weights[row],
                fd,
                fd_error,
            );
        }
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
        // The truth must NOT lie in the span of the 6-basis periodic B-spline.
        // A smooth low-frequency component alone would be fit to near machine
        // precision (σ² → 0), at which point the FD comparison cannot resolve
        // 1e-6 relative agreement against the analytic ∂(score)/∂y ∝ 1/dp. The
        // high-frequency component below leaves a genuine residual.
        let y = Array2::from_shape_fn((18, 2), |(row, col)| {
            let x = t[row];
            let phase = col as f64 + 1.0;
            0.1 + 0.4 * phase * x
                + (1.2 * x + 0.3 * phase).sin() * 0.25
                + 0.05 * (9.0 * x + 0.4 * phase).sin()
        });
        let knots = Array1::linspace(0.0, 1.0, 7);
        let penalty = Array2::from_diag(&array![0.0, 0.8, 1.1, 1.5, 2.0, 2.8]);
        let by = Array1::from_shape_fn(18, |row| 0.9 + 0.1 * (2.0 * t[row]).cos());
        let weights = Array1::from_shape_fn(18, |row| 1.0 + 0.06 * (1.4 * t[row]).sin());
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
    fn position_batched_duchon_forward_matches_prebuilt_design() {
        let t = Array1::linspace(0.05, 0.95, 16);
        let y = Array2::from_shape_fn((t.len(), 2), |(row, output)| {
            let u = t[row];
            let scale = output as f64 + 1.0;
            0.2 + 0.35 * scale * u + 0.12 * (5.0 * u + scale).sin()
        });
        let offsets = array![0_usize, 6_usize, 16_usize];
        let centers = Array1::linspace(0.0, 1.0, 6);
        let x = position_basis_design(t.view(), centers.view(), "duchon", 2, false, None)
            .expect("Duchon position basis");
        let penalty = Array2::from_diag(&Array1::from_shape_fn(x.ncols(), |col| {
            0.2 + 0.15 * col as f64
        }));

        let expected = gaussian_reml_fit_batched_impl(
            x.view(),
            y.view(),
            offsets.view(),
            penalty.view(),
            None,
            Some(0.8),
        )
        .expect("prebuilt Duchon batched fit");
        let actual = gaussian_reml_fit_positions_batched_impl(
            t.view(),
            y.view(),
            offsets.view(),
            centers.view(),
            "duchon",
            2,
            false,
            None,
            penalty.view(),
            None,
            Some(0.8),
            None,
            0,
        )
        .expect("streamed Duchon position batched fit");

        assert_eq!(actual.statuses, expected.statuses);
        for b in 0..2 {
            assert!((actual.lambdas[b] - expected.lambdas[b]).abs() < 1.0e-11);
            assert!((actual.reml_scores[b] - expected.reml_scores[b]).abs() < 1.0e-10);
        }
        for (actual, expected) in actual.coefficients.iter().zip(expected.coefficients.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-10);
        }
        for (actual, expected) in actual.fitted.iter().zip(expected.fitted.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-10);
        }
    }

    #[test]
    fn position_backward_grad_t_y_by_and_weight_match_finite_difference() {
        assert!(file!().ends_with(".rs"));
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
            0.0,
            Some(by.view()),
            0,
            None,
        )
        .expect("position analytic backward");
        let grad_by = backward.grad_by.expect("by gradient");
        let eps = 1.0e-5;

        for row in [2_usize, 8, 15] {
            let (fd, fd_error) = adaptive_finite_difference(t[row], eps, |candidate| {
                let mut perturbed = t.clone();
                perturbed[row] = candidate;
                position_objective(
                    perturbed.view(),
                    y.view(),
                    by.view(),
                    weights.view(),
                    knots.view(),
                    penalty.view(),
                    grad_lambda,
                    grad_coefficients.view(),
                    grad_fitted.view(),
                    grad_reml_score,
                )
            });
            assert_fd_estimate_close(
                &format!("position t[{row}]"),
                backward.grad_t[row],
                fd,
                fd_error,
            );
        }

        for (row, col) in [(1_usize, 0_usize), (10, 1)] {
            let (fd, fd_error) = adaptive_finite_difference(y[[row, col]], eps, |candidate| {
                let mut perturbed = y.clone();
                perturbed[[row, col]] = candidate;
                position_objective(
                    t.view(),
                    perturbed.view(),
                    by.view(),
                    weights.view(),
                    knots.view(),
                    penalty.view(),
                    grad_lambda,
                    grad_coefficients.view(),
                    grad_fitted.view(),
                    grad_reml_score,
                )
            });
            assert_fd_estimate_close(
                &format!("position y[{row},{col}]"),
                backward.grad_y[[row, col]],
                fd,
                fd_error,
            );
        }

        for row in [0_usize, 9, 17] {
            let (fd, fd_error) = adaptive_finite_difference(by[row], eps, |candidate| {
                let mut perturbed = by.clone();
                perturbed[row] = candidate;
                position_objective(
                    t.view(),
                    y.view(),
                    perturbed.view(),
                    weights.view(),
                    knots.view(),
                    penalty.view(),
                    grad_lambda,
                    grad_coefficients.view(),
                    grad_fitted.view(),
                    grad_reml_score,
                )
            });
            assert_fd_estimate_close(&format!("position by[{row}]"), grad_by[row], fd, fd_error);
        }

        for row in [1_usize, 7, 13] {
            let (fd, fd_error) = adaptive_finite_difference(weights[row], eps, |candidate| {
                let mut perturbed = weights.clone();
                perturbed[row] = candidate;
                position_objective(
                    t.view(),
                    y.view(),
                    by.view(),
                    perturbed.view(),
                    knots.view(),
                    penalty.view(),
                    grad_lambda,
                    grad_coefficients.view(),
                    grad_fitted.view(),
                    grad_reml_score,
                )
            });
            assert_fd_estimate_close(
                &format!("position weights[{row}]"),
                backward.grad_weights[row],
                fd,
                fd_error,
            );
        }
    }

    #[test]
    fn by_gate_backward_matches_forward_finite_difference_for_all_x_y_gate_and_weight_entries() {
        assert!(file!().ends_with(".rs"));
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
        let gated_x = apply_by_gate(x.view(), by.view(), 1).expect("by-gated design");
        let base_fit = gaussian_reml_multi_closed_form_with_cache(
            gated_x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.85),
            None,
        )
        .expect("base by-gated finite-difference fit");
        let fd_init_lambda = Some(base_fit.lambda);

        for target in targets {
            let fd_scale = 1.0_f64;
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
                    let (fd, fd_error) =
                        adaptive_finite_difference(x[[row, col]], eps, |candidate| {
                            let mut perturbed = x.clone();
                            perturbed[[row, col]] = candidate;
                            by_gate_objective(
                                perturbed.view(),
                                y.view(),
                                by.view(),
                                weights.view(),
                                penalty.view(),
                                target,
                                fd_init_lambda,
                            )
                        });
                    assert_fd_estimate_close(
                        &format!("target={target:?} x[{row},{col}]"),
                        grad_x[[row, col]],
                        fd_scale * fd,
                        fd_scale.abs() * fd_error,
                    );
                }
            }

            for row in 0..y.nrows() {
                for col in 0..y.ncols() {
                    let (fd, fd_error) =
                        adaptive_finite_difference(y[[row, col]], eps, |candidate| {
                            let mut perturbed = y.clone();
                            perturbed[[row, col]] = candidate;
                            by_gate_objective(
                                x.view(),
                                perturbed.view(),
                                by.view(),
                                weights.view(),
                                penalty.view(),
                                target,
                                fd_init_lambda,
                            )
                        });
                    assert_fd_estimate_close(
                        &format!("target={target:?} y[{row},{col}]"),
                        grad_y[[row, col]],
                        fd_scale * fd,
                        fd_scale.abs() * fd_error,
                    );
                }
            }

            for row in 0..by.len() {
                let (fd, fd_error) = adaptive_finite_difference(by[row], eps, |candidate| {
                    let mut perturbed = by.clone();
                    perturbed[row] = candidate;
                    by_gate_objective(
                        x.view(),
                        y.view(),
                        perturbed.view(),
                        weights.view(),
                        penalty.view(),
                        target,
                        fd_init_lambda,
                    )
                });
                assert_fd_estimate_close(
                    &format!("target={target:?} by[{row}]"),
                    grad_by[row],
                    fd_scale * fd,
                    fd_scale.abs() * fd_error,
                );
            }

            for row in 0..weights.len() {
                let (fd, fd_error) = adaptive_finite_difference(weights[row], eps, |candidate| {
                    let mut perturbed = weights.clone();
                    perturbed[row] = candidate;
                    by_gate_objective(
                        x.view(),
                        y.view(),
                        by.view(),
                        perturbed.view(),
                        penalty.view(),
                        target,
                        fd_init_lambda,
                    )
                });
                assert_fd_estimate_close(
                    &format!("target={target:?} weights[{row}]"),
                    grad_weights[row],
                    fd_scale * fd,
                    fd_scale.abs() * fd_error,
                );
            }
        }
    }

    #[test]
    fn batched_state_round_trip_matches_refit() {
        // Document the Task 3 state round-trip contract on the BATCHED
        // pyffi entry: backward called with `forward_state` set to the dict
        // returned by the matching forward must produce gradients that are
        // bit-exact identical to backward called without `forward_state`
        // (which refits internally). Guards against drift between
        // `_from_fit` and `_backward` for the batched path under any future
        // change to either branch.
        use ndarray::{array, s};

        let x = array![
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, -0.8],
            [1.0, -0.2],
            [1.0, 0.3],
            [1.0, 0.7],
            [1.0, 1.2],
        ];
        let y = array![
            [0.4, -0.2],
            [0.6, 0.1],
            [0.9, 0.3],
            [1.3, 0.4],
            [1.8, 0.6],
            [0.5, -0.1],
            [0.7, 0.2],
            [1.0, 0.3],
            [1.4, 0.5],
            [1.9, 0.7]
        ];
        let weights = array![1.0, 0.95, 1.05, 1.0, 0.98, 1.02, 0.99, 1.0, 1.01, 0.97];
        let row_offsets = array![0usize, 5, 10];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let init_lambda = Some(0.85_f64);

        let forward = gaussian_reml_fit_batched_impl(
            x.view(),
            y.view(),
            row_offsets.view(),
            penalty.view(),
            Some(weights.view()),
            init_lambda,
        )
        .expect("batched forward");
        let prebuilt_fits = (0..(row_offsets.len() - 1))
            .map(|b| {
                let start = row_offsets[b];
                let end = row_offsets[b + 1];
                let cache = gam::gaussian_reml::GaussianRemlEigenCache {
                    penalty_eigenvalues: forward
                        .cache_penalty_eigenvalues
                        .slice(s![b, ..])
                        .to_owned(),
                    eigenvectors: forward.cache_eigenvectors.slice(s![b, .., ..]).to_owned(),
                    coefficient_basis: forward
                        .cache_coefficient_basis
                        .slice(s![b, .., ..])
                        .to_owned(),
                    xtwx_fingerprint: forward.cache_xtwx_fingerprints[b],
                    penalty_fingerprint: forward.cache_penalty_fingerprints[b],
                    logdet_xtwx: forward.cache_logdet_xtwx[b],
                    logdet_penalty_positive: forward.cache_logdet_penalty_positive[b],
                    penalty_rank: forward.cache_penalty_ranks[b] as usize,
                    nullity: forward.cache_nullities[b] as usize,
                };
                Some(gam::gaussian_reml::GaussianRemlMultiResult {
                    lambda: forward.lambdas[b],
                    rho: forward.rhos[b],
                    coefficients: forward.coefficients.slice(s![b, .., ..]).to_owned(),
                    fitted: forward.fitted.slice(s![start..end, ..]).to_owned(),
                    reml_score: forward.reml_scores[b],
                    reml_grad_lambda: forward.reml_grad_lambdas[b],
                    reml_hess_lambda: forward.reml_hess_lambdas[b],
                    reml_grad_rho: forward.reml_grad_rhos[b],
                    reml_hess_rho: forward.reml_hess_rhos[b],
                    edf: forward.edf[b],
                    sigma2: forward.sigma2.slice(s![b, ..]).to_owned(),
                    cache,
                })
            })
            .collect::<Vec<_>>();

        let grad_lambda = array![0.2, -0.1];
        let grad_reml_score = array![-0.05, 0.08];

        let refit = gaussian_reml_fit_batched_backward_impl(
            x.view(),
            y.view(),
            row_offsets.view(),
            penalty.view(),
            Some(weights.view()),
            init_lambda,
            Some(grad_lambda.view()),
            None,
            None,
            Some(grad_reml_score.view()),
            None,
            None,
        )
        .expect("refit backward");
        let from_fits = gaussian_reml_fit_batched_backward_impl(
            x.view(),
            y.view(),
            row_offsets.view(),
            penalty.view(),
            Some(weights.view()),
            init_lambda,
            Some(grad_lambda.view()),
            None,
            None,
            Some(grad_reml_score.view()),
            None,
            Some(&prebuilt_fits),
        )
        .expect("from_fits backward");

        for (a, b) in refit.grad_x.iter().zip(from_fits.grad_x.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
        for (a, b) in refit.grad_y.iter().zip(from_fits.grad_y.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
        for (a, b) in refit.grad_weights.iter().zip(from_fits.grad_weights.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
    }
}
