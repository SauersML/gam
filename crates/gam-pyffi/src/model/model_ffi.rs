use competing_risks_decode::{
    competing_risks_columns, competing_risks_numeric_list, competing_risks_string_list,
    set_optional_competing_risks_matrix, set_optional_competing_risks_vector,
};

use manifold_pyclasses::{
    CircleManifold, EuclideanManifold, GrassmannManifold, ProductManifold, SpdManifold,
    SphereManifold, StiefelManifold, TorusManifold,
};

use python_literal::{python_float_display, python_string_repr};

use sklearn_metadata::sklearn_fit_metadata;

use summary_render::{summary_html_escape, summary_render_coefficients_html, summary_render_value};

use survival_surface_io::{
    hazard_from_cumulative_knots, interpolate_rows, survival_block,
    survival_block_cumulative_hazard, survival_block_failure, survival_block_hazard,
    survival_chunk_defaults, survival_chunk_iter_collect, survival_chunk_ranges,
    survival_coerce_times, survival_cumulative_from_survival, survival_failure_from_survival,
    survival_ffi_surface, survival_parameters_matrix, survival_should_chunk, write_survival_csv,
};

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
    /// Single uncertainty knob (issue #342): `Some(level)` requests the
    /// full-uncertainty predictor with that pointwise coverage level
    /// (yields `std_error` and CI bounds on the standard path; per-cell
    /// `survival_se` / `eta_se` on the survival path). `None` requests
    /// point predictions only. There is no separate `with_uncertainty`
    /// flag — coverage and the request to quantify uncertainty are the
    /// same decision. (Issue #310 renamed the SE column from
    /// `effective_se` and dropped the redundant `effective_variance` ==
    /// `std_error ** 2` column.)
    interval: Option<f64>,
    time_grid: Option<Vec<f64>>,
    /// Posterior covariance source for eta/mean intervals. One of
    /// `"conditional"` (H⁻¹ only), `"smoothing"` (first-order smoothing
    /// correction `H⁻¹ + J Var(ρ̂) Jᵀ` when available, else falls back to
    /// conditional), or `"required"` (the smoothing correction, erroring if it
    /// is unavailable). `None` keeps the engine default (`"smoothing"`). This
    /// is the Python/CLI parity surface for `--covariance-mode`; it is read on
    /// the delta-method (effectively-linear + interval) predict branch where
    /// `PredictUncertaintyOptions` governs the covariance, mirroring the CLI's
    /// `gam predict --covariance-mode`.
    #[serde(default)]
    covariance_mode: Option<String>,
    /// When `true`, the effectively-linear interval branch also returns
    /// response-scale observation intervals `Var(y_new|x) = Var(μ̂) + Var(Y|μ)`
    /// via the engine's `includeobservation_interval`, surfaced as
    /// `observation_lower` / `observation_upper` columns. `None`/`false`
    /// preserves the prior behaviour (no observation interval).
    #[serde(default)]
    observation_interval: Option<bool>,
    /// Opt-in distribution-free conformal calibration of the response-scale
    /// interval (issue #310 family path). When `Some(level)` with
    /// `level ∈ (0, 1)`, the model-based `mean_lower` / `mean_upper` are
    /// REPLACED by the split-conformal interval `μ̂(x) ± q̂·s(x)` calibrated
    /// from a held-out fold supplied via the `*_conformal` predict pyfunction.
    /// `None` (default) leaves the interval untouched. Only the conformal
    /// predict path reads this field.
    #[serde(default)]
    conformal_level: Option<f64>,
}

/// Parse the public `covariance_mode` string into the engine enum. `None`
/// keeps the engine default (smoothing-preferred); unknown strings are a hard
/// error so a typo never silently degrades to the default covariance.
fn parse_covariance_mode(
    raw: Option<&str>,
) -> Result<Option<gam_predict::InferenceCovarianceMode>, String> {
    let Some(text) = raw else {
        return Ok(None);
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "conditional" => Ok(Some(gam_predict::InferenceCovarianceMode::Conditional)),
        "smoothing" => Ok(Some(
            gam_predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
        )),
        "required" => Ok(Some(
            gam_predict::InferenceCovarianceMode::ConditionalPlusSmoothingRequired,
        )),
        other => Err(format!(
            "covariance_mode must be one of \"conditional\", \"smoothing\", or \"required\"; \
             got \"{other}\""
        )),
    }
}

#[derive(Serialize)]
struct PyPredictOptionsPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    interval: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_grid: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observation_interval: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    conformal_level: Option<f64>,
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

/// Per-smooth significance row for the FFI summary — the canonical mgcv
/// `summary.gam` smooth-term table (`edf`, reference d.f., test statistic, and
/// p-value). Random-effect smooths report only `edf` (their boundary
/// variance-component test is not a Wald χ²); penalized smooth terms carry the
/// Wood (2013) rank-truncated Wald `chi_sq` / `p_value`. The shape mirrors the
/// CLI's `SmoothTermSummary`.
///
/// This `p_value` is the *first-order* Wald reference. The summary table is
/// built from a saved model without the training rows, so it cannot run the
/// per-term constrained refits the second-order test needs. The
/// **second-order-accurate, Bartlett-corrected likelihood-ratio** p-value is
/// computed on demand by `smooth_term_lr_inference_json` (Python
/// `Model.smooth_significance(data)`), which auto-applies the exact Lawley
/// factor whenever the family carries closed-form cumulant jets (#939/#1063).
#[derive(Serialize)]
struct SummarySmoothTermRow {
    name: String,
    edf: f64,
    ref_df: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    chi_sq: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p_value: Option<f64>,
}

/// The fitted curvature estimate for one `curv(...)` constant-curvature smooth
/// (#944). κ̂ is read directly off the resolved (fitted) basis spec — the
/// outer optimiser's argmin of the profiled criterion over κ — so it is an
/// estimate the fit ALREADY produced and is surfaced with zero refit. The
/// profile-CI and the interior κ = 0 flatness LR test require re-profiling
/// `V_p(κ)` against the original data and are produced on demand by the
/// `curvature_inference_json` entry (which the data-carrying caller invokes).
#[derive(Serialize)]
struct SummaryCurvatureRow {
    /// `curv(...)` term name.
    name: String,
    /// Smooth-term index of the constant-curvature term.
    term_idx: usize,
    /// Fitted signed sectional curvature κ̂.
    kappa_hat: f64,
    /// Sign-of-κ̂ geometry tag: `"spherical"` (κ̂>0), `"flat"` (κ̂≈0), or
    /// `"hyperbolic"` (κ̂<0). A point estimate only — the level-α verdict comes
    /// from the profile-CI endpoints via `curvature_inference_json`.
    geometry: &'static str,
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
    /// Log-likelihood at the converged mode (engine constants-omitted scale).
    /// Carried so `compare_models` can form the Occam-penalised conditional AIC
    /// it ranks on (issue #1362). Optional for forward/backward compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    log_likelihood: Option<f64>,
    /// Number of observations the fit was trained on. Carried so `compare_models`
    /// can REFUSE to rank fits made on different-sized (hence different) data:
    /// `−2·loglik` / REML evidence grow with `n`, so a score gap between two fits
    /// with different `n` is not a Bayes factor (#1384 sibling — the same
    /// fail-loud contract as the family guard). Sourced from the IRLS working-set
    /// length; `None` for fit paths that do not retain the working geometry (e.g.
    /// O(n) spline-scan models), which the guard then treats as unconstrained.
    #[serde(skip_serializing_if = "Option::is_none")]
    n_obs: Option<usize>,
    reml_score: f64,
    raw_reml_score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    null_space_logdet: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    null_dim: Option<f64>,
    iterations: usize,
    edf_total: Option<f64>,
    lambdas: Vec<f64>,
    coefficients: Vec<SummaryCoefficientRow>,
    /// Per-smooth significance table (mgcv-style). Empty when the model has no
    /// smooth/random-effect terms or when the design could not be reconstructed
    /// to recover the per-term coefficient blocks (e.g. a model saved without
    /// `resolved_termspec` / training feature ranges).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    smooth_terms: Vec<SummarySmoothTermRow>,
    /// Fitted curvature estimates for any `curv(...)` constant-curvature smooths
    /// (#944). Empty when the model has no constant-curvature term. κ̂ is the
    /// estimate the fit already produced; the CI and flatness p-value are
    /// produced on demand by `curvature_inference_json`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    curvature_estimands: Vec<SummaryCurvatureRow>,
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
    /// Predictive-class discriminator (e.g. "standard", "transformation-normal",
    /// "bernoulli marginal-slope"). The Python `shape_predict_response` dispatcher
    /// reads this to pick the right shaper, exactly as the survival payload's
    /// `model_class` does. Standard models historically omitted it, which broke
    /// `Model.predict()` with `KeyError: 'model_class'` once the defensive
    /// `parsed.get(...)` fallback was removed from the post-shim shaper (#866/#867).
    model_class: String,
    /// Inverse-link family kind tag (`identity`, `logit`, `probit`, `log`, ...).
    /// The shaper consults it alongside `model_class` to disambiguate the
    /// Bernoulli marginal-slope path from the survival marginal-slope variant.
    family: String,
    /// Provenance of the returned prediction interval (#942). Present only when
    /// an interval was requested. `"jackknife+ (distribution-free, finite-sample
    /// ≥level coverage)"` when the exact Gaussian-identity jackknife+ magic ran;
    /// `"model-based (Gaussian posterior)"` when the eligibility gate fell back
    /// to the model's credible/predictive band. `None` (omitted) for point-only
    /// predictions or model classes that do not carry the field yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    interval_method: Option<String>,
}

/// Typed wire payload for NUTS posterior draws.
///
/// Bulk numeric fields stay as ndarrays until the PyO3 edge, where they become
/// NumPy arrays. Only names and scalar diagnostics become ordinary Python
/// objects; no draw-sized JSON or Python list is materialized.
struct SamplePayload {
    samples: Array2<f64>,
    coefficient_names: Vec<String>,
    posterior_mean: Array1<f64>,
    posterior_std: Array1<f64>,
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
    /// Serialized parameterized [`gam::types::InverseLink`] (JSON). The bare
    /// `family_kind` tag cannot represent the per-fit state of the
    /// parameterized links (`Sas`, `Mixture`, `LatentCLogLog`, `BetaLogistic`),
    /// so this carries the full link spec back into the response-scale
    /// transforms. `None` when serialization is unavailable; the wrapper then
    /// falls back to the string tag (issue #1133).
    link_spec: Option<String>,
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

// Every f64-bearing field below routes through `finite_safe_json` so that the
// non-finite values a survival surface can legitimately carry (`+∞` cumulative
// hazard / hazard in a saturated tail, or a genuine `NaN` worth surfacing for
// debugging) survive the JSON round-trip instead of degrading to a bare `null`
// that the typed deserializer then rejects (#1564). The serialize and
// deserialize structs use the SAME adapters, so the wire format is symmetric.
#[derive(Serialize)]
struct SurvivalPredictionPayload {
    class: &'static str,
    model_class: String,
    likelihood_mode: String,
    #[serde(with = "crate::finite_safe_json::vec")]
    times: Vec<f64>,
    #[serde(with = "crate::finite_safe_json::matrix")]
    hazard: Vec<Vec<f64>>,
    #[serde(with = "crate::finite_safe_json::matrix")]
    survival: Vec<Vec<f64>>,
    #[serde(with = "crate::finite_safe_json::matrix")]
    cumulative_hazard: Vec<Vec<f64>>,
    #[serde(with = "crate::finite_safe_json::vec")]
    linear_predictor: Vec<f64>,
    #[serde(with = "crate::finite_safe_json::map")]
    columns: BTreeMap<String, Vec<f64>>,
    /// Delta-method standard errors on the survival surface, when the
    /// caller requested uncertainty via `interval=...`.  Same shape as
    /// `survival`.  `None` otherwise.
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "crate::finite_safe_json::opt_matrix"
    )]
    survival_se: Option<Vec<Vec<f64>>>,
    /// Delta-method SE on the linear predictor at each row's own exit
    /// time, when uncertainty was requested.  Length equals
    /// `linear_predictor.len()`.
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "crate::finite_safe_json::opt_vec"
    )]
    eta_se: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct SurvivalPredictionJsonPayload {
    class: String,
    model_class: Option<String>,
    #[serde(default, with = "crate::finite_safe_json::opt_vec")]
    times: Option<Vec<f64>>,
    #[serde(default, with = "crate::finite_safe_json::opt_matrix")]
    hazard: Option<Vec<Vec<f64>>>,
    #[serde(default, with = "crate::finite_safe_json::opt_matrix")]
    survival: Option<Vec<Vec<f64>>>,
    #[serde(default, with = "crate::finite_safe_json::opt_matrix")]
    cumulative_hazard: Option<Vec<Vec<f64>>>,
    #[serde(default, with = "crate::finite_safe_json::opt_vec")]
    linear_predictor: Option<Vec<f64>>,
    #[serde(default, with = "crate::finite_safe_json::opt_map")]
    columns: Option<BTreeMap<String, Vec<f64>>>,
    #[serde(default, with = "crate::finite_safe_json::opt_matrix")]
    survival_se: Option<Vec<Vec<f64>>>,
    #[serde(default, with = "crate::finite_safe_json::opt_vec")]
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

// The gamfit exception hierarchy and the typed engine→Python error
// adaptors now live in `crate::ffi_errors`; they are re-exported at the
// crate root so the `#[pyfunction]`s below keep referring to them by their
// bare names (`py_value_error`, `estimation_error_to_pyerr`, etc.).

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
            "predict_conformal",
            "build_predict_payload_json",
            "interpolate_rows",
            "survival_chunk_defaults",
            "survival_chunk_ranges",
            "survival_should_chunk",
            "survival_chunk_iter_collect",
            "write_survival_csv",
            "default_survival_time_grid",
            "competing_risks_cif",
            "competing_risks_cif_from_predictions",
            "survival_prediction_payload_from_json",
            "competing_risks_prediction_payload_from_json",
            "sample",
            "summary",
            "summary_html",
            "check",
            "report",
            "sklearn_fit_metadata",
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
            "gaussian_weighted_ridge_batch_backward",
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
            "response_geometry_ilr",
            "response_geometry_inverse_ilr",
            "response_geometry_aitchison_metric",
            "response_geometry_clr_jet",
            "response_geometry_simplex_log_map_jet",
            "response_geometry_simplex_exp_map_jet",
            "response_geometry_sphere_exp_map_jet",
            "response_geometry_fit_curvature",
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
    // SAE row-block analytic penalty kinds this build supports, so the Python
    // wrapper can refuse cleanly on a stale extension rather than forward a
    // descriptor that fails with a cryptic Schur error (issue #338). Derived
    // from the same source of truth as `sae_penalty_is_row_block_supported`.
    info.set_item(
        "sae_row_block_penalties",
        gam::terms::sae::manifold::sae_row_block_penalty_kinds().to_vec(),
    )?;
    Ok(info.unbind())
}

/// Rust-owned typed table used by every named-table Python entry point.
///
/// The old boundary accepted `Vec<Vec<String>>`, forcing numeric cells through
/// Python objects and then reparsing their string representations. This class
/// owns the canonical `EncodedDataset`. Its sequence protocol renders only a
/// requested row for the few metadata helpers that still consume text.
#[pyclass(name = "_EncodedTable", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyEncodedTable {
    dataset: EncodedDataset,
}

impl PyEncodedTable {
    fn require_headers(&self, headers: &[String]) -> Result<(), String> {
        if self.dataset.headers == headers {
            Ok(())
        } else {
            Err(format!(
                "encoded table headers {:?} do not match call headers {:?}",
                self.dataset.headers, headers
            ))
        }
    }

    fn rendered_row(&self, row: usize) -> Result<Vec<String>, String> {
        if row >= self.dataset.values.nrows() {
            return Err(format!(
                "encoded table row {row} is outside 0..{}",
                self.dataset.values.nrows()
            ));
        }
        self.dataset
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(column, schema)| {
                let value = self.dataset.values[[row, column]];
                match schema.kind {
                    ColumnKindTag::Categorical => {
                        let code = value as usize;
                        if value < 0.0 || value.fract() != 0.0 || code >= schema.levels.len() {
                            return Err(format!(
                                "categorical column '{}' has invalid encoded value {value} at row {}",
                                schema.name,
                                row + 1
                            ));
                        }
                        // Legacy table entry points still infer from strings. Mark
                        // this one lazily-rendered row so numeric-looking labels
                        // retain their categorical source intent.
                        Ok(format!(
                            "{}{}",
                            gam::data::CATEGORICAL_CELL_SENTINEL,
                            schema.levels[code]
                        ))
                    }
                    ColumnKindTag::Binary => Ok(if value == 0.0 {
                        "0".to_string()
                    } else {
                        "1".to_string()
                    }),
                    ColumnKindTag::Continuous => Ok(format!("{value:?}")),
                }
            })
            .collect()
    }
}

#[pymethods]
impl PyEncodedTable {
    fn __len__(&self) -> usize {
        self.dataset.values.nrows()
    }

    fn __getitem__(&self, index: isize) -> PyResult<Vec<String>> {
        let n = self.dataset.values.nrows() as isize;
        let normalized = if index < 0 { index + n } else { index };
        if normalized < 0 || normalized >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "encoded table row index {index} out of range for {n} rows"
            )));
        }
        self.rendered_row(normalized as usize)
            .map_err(py_value_error)
    }

    #[getter]
    fn headers(&self) -> Vec<String> {
        self.dataset.headers.clone()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.dataset.values.dim()
    }

    fn __repr__(&self) -> String {
        let (rows, columns) = self.dataset.values.dim();
        format!("_EncodedTable(n_rows={rows}, n_columns={columns})")
    }
}

fn validate_column_partition(
    n_columns: usize,
    numeric_positions: &[usize],
    categorical_positions: &[usize],
) -> Result<(), String> {
    let mut seen = vec![false; n_columns];
    for (kind, positions) in [
        ("numeric", numeric_positions),
        ("categorical", categorical_positions),
    ] {
        for &position in positions {
            if position >= n_columns {
                return Err(format!(
                    "{kind} column position {position} is outside 0..{n_columns}"
                ));
            }
            if std::mem::replace(&mut seen[position], true) {
                return Err(format!("column position {position} is repeated"));
            }
        }
    }
    if let Some(position) = seen.iter().position(|present| !present) {
        return Err(format!(
            "column position {position} is missing from the numeric/categorical partition"
        ));
    }
    Ok(())
}

/// Construct an encoded table from one homogeneous numeric matrix plus the
/// genuinely categorical string columns. Numeric cells cross through
/// rust-numpy; only categorical labels are Python strings.
#[pyfunction]
fn encoded_table_from_columns(
    headers: Vec<String>,
    numeric_values: PyReadonlyArray2<'_, f64>,
    numeric_positions: Vec<usize>,
    categorical_values: Vec<Vec<String>>,
    categorical_positions: Vec<usize>,
) -> PyResult<PyEncodedTable> {
    ensure_unique_headers(&headers).map_err(py_value_error)?;
    validate_column_partition(headers.len(), &numeric_positions, &categorical_positions)
        .map_err(py_value_error)?;
    let numeric = numeric_values.as_array();
    if numeric.ncols() != numeric_positions.len() {
        return Err(py_value_error(format!(
            "numeric matrix has {} columns but {} numeric positions were supplied",
            numeric.ncols(),
            numeric_positions.len()
        )));
    }
    if categorical_values.len() != categorical_positions.len() {
        return Err(py_value_error(format!(
            "received {} categorical columns but {} categorical positions",
            categorical_values.len(),
            categorical_positions.len()
        )));
    }
    let n_rows = if !numeric_positions.is_empty() {
        numeric.nrows()
    } else {
        categorical_values.first().map(Vec::len).unwrap_or(0)
    };
    if n_rows == 0 {
        return Err(py_value_error("table data cannot be empty".to_string()));
    }
    for (index, column) in categorical_values.iter().enumerate() {
        if column.len() != n_rows {
            return Err(py_value_error(format!(
                "categorical column '{}' has {} rows but expected {n_rows}",
                headers[categorical_positions[index]],
                column.len()
            )));
        }
    }

    let mut values = Array2::<f64>::zeros((n_rows, headers.len()));
    let mut schema_columns = vec![None::<SchemaColumn>; headers.len()];
    let mut column_kinds = vec![ColumnKindTag::Continuous; headers.len()];
    for (matrix_column, &table_column) in numeric_positions.iter().enumerate() {
        let column = numeric.column(matrix_column);
        if let Some(row) = column.iter().position(|value| !value.is_finite()) {
            return Err(py_value_error(format!(
                "non-finite value at row {}, column '{}'",
                row + 1,
                headers[table_column]
            )));
        }
        let kind = infer_numeric_array_column_kind(column);
        values.column_mut(table_column).assign(&column);
        column_kinds[table_column] = kind;
        schema_columns[table_column] = Some(SchemaColumn {
            name: headers[table_column].clone(),
            kind,
            levels: Vec::new(),
        });
    }
    for (source_column, &table_column) in
        categorical_values.iter().zip(categorical_positions.iter())
    {
        let marked = source_column
            .iter()
            .map(|value| format!("{}{}", gam::data::CATEGORICAL_CELL_SENTINEL, value))
            .collect::<Vec<_>>();
        let refs = marked.iter().map(String::as_str).collect::<Vec<_>>();
        let (schema, encoded) =
            infer_and_encode_column_major(&headers[table_column], &refs, table_column + 1)
                .map_err(py_value_error)?;
        values
            .column_mut(table_column)
            .assign(&ndarray::ArrayView1::from(&encoded));
        column_kinds[table_column] = schema.kind;
        schema_columns[table_column] = Some(schema);
    }
    let schema_columns = schema_columns
        .into_iter()
        .enumerate()
        .map(|(column, schema)| schema.ok_or_else(|| format!("missing schema for column {column}")))
        .collect::<Result<Vec<_>, _>>()
        .map_err(py_value_error)?;
    Ok(PyEncodedTable {
        dataset: EncodedDataset {
            headers,
            values,
            schema: DataSchema {
                columns: schema_columns,
            },
            column_kinds,
        },
    })
}

/// Import a pandas/Polars/PyArrow provider through the Arrow C Stream
/// PyCapsule protocol. The producer owns its buffers until arrow-rs consumes
/// the stream; numeric primitives are read directly and only string columns
/// allocate level labels.
#[pyfunction]
fn encoded_table_from_arrow(
    headers: Vec<String>,
    source: &Bound<'_, PyAny>,
) -> PyResult<PyEncodedTable> {
    use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
    use pyo3::types::{PyCapsule, PyCapsuleMethods};

    let capsule = source
        .getattr("__arrow_c_stream__")?
        .call0()?
        .cast_into::<PyCapsule>()?;
    let stream_pointer = capsule
        .pointer_checked(Some(c"arrow_array_stream"))?
        .cast::<FFI_ArrowArrayStream>()
        .as_ptr();
    // SAFETY: the capsule name is validated above and the Arrow PyCapsule
    // protocol guarantees a writable, initialized FFI_ArrowArrayStream. The
    // move nulls the capsule's release callback, so ownership is unique.
    let stream = unsafe { FFI_ArrowArrayStream::from_raw(stream_pointer) };
    let mut reader = ArrowArrayStreamReader::try_new(stream)
        .map_err(|error| py_value_error(format!("failed to import Arrow C stream: {error}")))?;
    let dataset =
        gam::data::encode_arrow_record_batch_reader_with_inferred_schema(&mut reader, headers)
            .map_err(|error| py_value_error(error.to_string()))?;
    Ok(PyEncodedTable { dataset })
}

/// Project and re-encode a typed prediction table against a saved model schema.
///
/// Both sides are already numeric/categorical columns, so matching a category
/// means mapping its source level label to the frozen training-level code. No
/// cell is rendered to a string and numeric columns are copied exactly once
/// into the projected dense matrix.
fn dataset_with_model_schema_from_encoded(
    model: &FittedModel,
    source: &EncodedDataset,
) -> Result<EncodedDataset, String> {
    let required = required_prediction_columns(model)?;
    let present = source.headers.iter().cloned().collect::<BTreeSet<_>>();
    let missing = required
        .difference(&present)
        .map(|name| format!("missing required column '{name}'"))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(missing.join(" "));
    }
    let consumable = prediction_consumable_columns(model)?;
    let mut keep = source
        .headers
        .iter()
        .enumerate()
        .filter(|(_, name)| consumable.contains(name.as_str()))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    // Preserve row count for intercept-only models, exactly like the textual
    // projection path: a zero-column dataset cannot represent N.
    if keep.is_empty() {
        keep = (0..source.headers.len()).collect();
    }

    let training_schema = model.require_data_schema()?;
    let training_by_name = training_schema
        .columns
        .iter()
        .map(|column| (column.name.as_str(), column))
        .collect::<HashMap<_, _>>();
    let encode_unknown = model.random_effect_group_columns();
    let n_rows = source.values.nrows();
    let mut headers = Vec::with_capacity(keep.len());
    let mut values = Array2::<f64>::zeros((n_rows, keep.len()));
    let mut schema_columns = Vec::with_capacity(keep.len());
    let mut column_kinds = Vec::with_capacity(keep.len());

    for (destination, &source_index) in keep.iter().enumerate() {
        let name = &source.headers[source_index];
        let source_schema = source
            .schema
            .columns
            .get(source_index)
            .ok_or_else(|| format!("encoded table column '{name}' has no source schema"))?;
        let target_schema = training_by_name
            .get(name.as_str())
            .ok_or_else(|| format!("column '{name}' was not present in the training schema"))?;
        match (source_schema.kind, target_schema.kind) {
            (ColumnKindTag::Categorical, ColumnKindTag::Categorical) => {
                let target_levels = target_schema
                    .levels
                    .iter()
                    .enumerate()
                    .map(|(index, level)| (level.as_str(), index))
                    .collect::<HashMap<_, _>>();
                for row in 0..n_rows {
                    let raw_code = source.values[[row, source_index]];
                    let code = raw_code as usize;
                    if raw_code < 0.0
                        || raw_code.fract() != 0.0
                        || code >= source_schema.levels.len()
                    {
                        return Err(format!(
                            "categorical column '{name}' has invalid encoded value {raw_code} at row {}",
                            row + 1
                        ));
                    }
                    let label = &source_schema.levels[code];
                    values[[row, destination]] = match target_levels.get(label.as_str()) {
                        Some(index) => *index as f64,
                        None if encode_unknown.contains(name) => target_schema.levels.len() as f64,
                        None => {
                            return Err(format!(
                                "unseen level '{}' in categorical column '{}' at row {}",
                                label,
                                name,
                                row + 1
                            ));
                        }
                    };
                }
            }
            (ColumnKindTag::Categorical, _) => {
                return Err(format!(
                    "column '{name}' is categorical in prediction data but numeric in the training schema"
                ));
            }
            (_, ColumnKindTag::Categorical) => {
                return Err(format!(
                    "column '{name}' is numeric in prediction data but categorical in the training schema"
                ));
            }
            (_, ColumnKindTag::Binary) => {
                for row in 0..n_rows {
                    let value = source.values[[row, source_index]];
                    if (value - 0.0).abs() >= 1e-12 && (value - 1.0).abs() >= 1e-12 {
                        return Err(format!(
                            "column '{name}' is binary in schema but row {} has value {value}; expected 0 or 1",
                            row + 1
                        ));
                    }
                    values[[row, destination]] = value;
                }
            }
            (_, ColumnKindTag::Continuous) => {
                values
                    .column_mut(destination)
                    .assign(&source.values.column(source_index));
            }
        }
        headers.push(name.clone());
        schema_columns.push((*target_schema).clone());
        column_kinds.push(target_schema.kind);
    }
    Ok(EncodedDataset {
        headers,
        values,
        schema: DataSchema {
            columns: schema_columns,
        },
        column_kinds,
    })
}

fn schema_check_encoded(
    model: &FittedModel,
    source: &EncodedDataset,
) -> Result<SchemaCheckPayload, String> {
    let expected = required_prediction_columns(model)?;
    let present = source.headers.iter().cloned().collect::<BTreeSet<_>>();
    let mut issues = expected
        .difference(&present)
        .map(|missing| SchemaIssue {
            kind: "missing_column".to_string(),
            message: format!("missing required column '{missing}'"),
            column: Some(missing.clone()),
        })
        .collect::<Vec<_>>();
    if issues.is_empty()
        && let Err(message) = dataset_with_model_schema_from_encoded(model, source)
    {
        issues.push(SchemaIssue {
            kind: "schema_error".to_string(),
            message,
            column: None,
        });
    }
    if issues.is_empty() {
        for (column, vocabulary) in model.numeric_fixed_factor_vocabularies() {
            let Some(index) = source.headers.iter().position(|header| header == &column) else {
                continue;
            };
            for value in source.values.column(index) {
                if !vocabulary.contains(&gam::data::canonical_level_bits(*value)) {
                    issues.push(SchemaIssue {
                        kind: "schema_error".to_string(),
                        message: format!(
                            "unseen level '{value}' in fixed factor column '{column}'; \
                             the factor's levels were fixed at fit time"
                        ),
                        column: Some(column.clone()),
                    });
                    break;
                }
            }
        }
    }
    Ok(SchemaCheckPayload {
        ok: issues.is_empty(),
        issues,
    })
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

    array.cast_into::<PyArray2<f64>>().map_err(PyErr::from)
}

#[pyfunction]
fn numeric_matrix_f64<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    label: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float")?;
    let array = np.call_method("asarray", (values,), Some(&kwargs))?;
    numeric_matrix_validate(py, &array, label)
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
    if let Some(values) = columns.get("linear_predictor") {
        return Ok(values.clone());
    }
    Err(PyKeyError::new_err(
        "transformation-normal prediction payload is missing linear_predictor",
    ))
}

#[pyfunction]
fn column_stack_f64<'py>(py: Python<'py>, columns: Vec<Vec<f64>>) -> PyResult<Py<PyArray2<f64>>> {
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
    let out = Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|err| py_value_error(format!("failed to reshape design matrix: {err}")))?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn vec_to_array1_f64<'py>(py: Python<'py>, values: Vec<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Ok(Array1::from_vec(values).into_pyarray(py).unbind())
}

fn survival_prediction_matrix_from_rows(rows: Vec<Vec<f64>>, label: &str) -> PyResult<Array2<f64>> {
    if rows.is_empty() {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != n_cols {
            return Err(py_value_error(format!(
                "{label} row {row_idx} has length {} but expected {n_cols}",
                row.len()
            )));
        }
    }
    let data = rows.into_iter().flatten().collect::<Vec<_>>();
    Array2::from_shape_vec((n_rows, n_cols), data)
        .map_err(|err| py_value_error(format!("failed to reshape {label}: {err}")))
}

fn survival_prediction_parameters_from_columns(
    columns: &BTreeMap<String, Vec<f64>>,
    linear_predictor: &[f64],
) -> PyResult<Array2<f64>> {
    if columns.is_empty() {
        if linear_predictor.is_empty() {
            return Ok(Array2::<f64>::zeros((0, 0)));
        }
        return Array2::from_shape_vec((linear_predictor.len(), 1), linear_predictor.to_vec())
            .map_err(|err| {
                py_value_error(format!("failed to reshape survival parameters: {err}"))
            });
    }

    let n_rows = columns.values().next().map(Vec::len).unwrap_or(0);
    let n_cols = columns.len();
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for (col_idx, (name, values)) in columns.iter().enumerate() {
        if values.len() != n_rows {
            return Err(py_value_error(format!(
                "survival parameter column '{name}' has length {} but expected {n_rows}",
                values.len()
            )));
        }
        for (row_idx, value) in values.iter().enumerate() {
            out[[row_idx, col_idx]] = *value;
        }
    }
    Ok(out)
}

fn set_survival_prediction_array1<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    key: &str,
    values: Vec<f64>,
) -> PyResult<()> {
    if values.is_empty() {
        out.set_item(key, py.None())
    } else {
        out.set_item(key, Array1::from_vec(values).into_pyarray(py))
    }
}

fn set_survival_prediction_matrix<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    key: &str,
    rows: Option<Vec<Vec<f64>>>,
) -> PyResult<()> {
    match rows {
        Some(values) => out.set_item(
            key,
            survival_prediction_matrix_from_rows(values, key)?.into_pyarray(py),
        ),
        None => out.set_item(key, py.None()),
    }
}

#[pyfunction]
fn survival_prediction_payload_from_json(py: Python<'_>, raw: &str) -> PyResult<PyObject> {
    let payload: SurvivalPredictionJsonPayload = serde_json::from_str(raw).map_err(|err| {
        py_value_error(format!(
            "failed to parse survival prediction payload: {err}"
        ))
    })?;
    if payload.class != "survival_prediction" {
        return Err(py_value_error(format!(
            "expected survival_prediction payload, got '{}'",
            payload.class
        )));
    }

    let out = PyDict::new(py);
    let model_class = payload
        .model_class
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "survival marginal-slope".to_string());
    out.set_item("model_class", model_class)?;

    set_survival_prediction_array1(py, &out, "times", payload.times.unwrap_or_default())?;
    set_survival_prediction_matrix(py, &out, "hazard", payload.hazard)?;
    set_survival_prediction_matrix(py, &out, "survival", payload.survival)?;
    set_survival_prediction_matrix(py, &out, "cumulative_hazard", payload.cumulative_hazard)?;

    let linear_predictor = payload.linear_predictor.unwrap_or_default();
    set_survival_prediction_array1(py, &out, "linear_predictor", linear_predictor.clone())?;
    set_survival_prediction_matrix(py, &out, "survival_se", payload.survival_se)?;
    set_survival_prediction_array1(py, &out, "eta_se", payload.eta_se.unwrap_or_default())?;

    let columns = payload.columns.unwrap_or_default();
    let parameter_names = columns.keys().cloned().collect::<Vec<_>>();
    out.set_item("parameter_names", PyTuple::new(py, parameter_names)?)?;
    out.set_item(
        "parameters",
        survival_prediction_parameters_from_columns(&columns, &linear_predictor)?.into_pyarray(py),
    )?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
fn competing_risks_prediction_payload_from_json(py: Python<'_>, raw: &str) -> PyResult<PyObject> {
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

#[pyfunction]
fn extract_row_ids(
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
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
    rows.require_headers(&headers).map_err(py_value_error)?;
    let schema =
        rows.dataset.schema.columns.get(index).ok_or_else(|| {
            py_value_error(format!("id_column '{id_column}' has no encoded schema"))
        })?;
    let mut row_ids = Vec::with_capacity(rows.dataset.values.nrows());
    for row in 0..rows.dataset.values.nrows() {
        let value = rows.dataset.values[[row, index]];
        row_ids.push(match schema.kind {
            ColumnKindTag::Categorical => {
                let code = value as usize;
                schema.levels.get(code).cloned().ok_or_else(|| {
                    py_value_error(format!(
                        "id_column '{id_column}' has invalid category code {value} at row {}",
                        row + 1
                    ))
                })?
            }
            ColumnKindTag::Binary => if value == 0.0 { "0" } else { "1" }.to_string(),
            ColumnKindTag::Continuous => format!("{value:?}"),
        });
    }
    Ok(Some(row_ids))
}

/// Training-time upper bound for the default survival surface grid, read from
/// the saved model rather than the prediction frame.
///
/// The default surface grid must be a property of the FITTED model, not of the
/// `exit` placeholder a caller happens to put in the prediction frame. A small
/// placeholder `exit` previously shrank the grid to `[entry, exit]` and silently
/// truncated the surface, so `survival_at(t)` for an ordinary in-fitted-range
/// `t` past that placeholder fell through to the `t -> inf` asymptote (`S = 0`)
/// — issue #896. Anchoring the grid's upper edge to the training time support
/// keeps every in-range query time inside the surface regardless of the
/// placeholder.
///
/// Sources, in order:
///   * `survival_time_knots` (bspline / ispline bases) live on the `log(t)`
///     axis spanning the training entry/exit range, so `exp(max knot)` is the
///     largest training time the basis was fit over.
///   * `survival_baseline_scale` (the linear Weibull basis stores no knots) is
///     the Weibull characteristic time; a few multiples of it comfortably cover
///     the fitted range, which is all a grid UPPER bound needs.
///
/// Returns `None` when the model carries neither (the caller then falls back to
/// the prediction-frame range alone, preserving the prior behavior).
fn saved_survival_training_time_upper_bound(model_bytes: &[u8]) -> Option<f64> {
    let saved: serde_json::Value = serde_json::from_slice(model_bytes).ok()?;
    let payload = saved
        .get("payload")
        .and_then(serde_json::Value::as_object)?;

    if let Some(knots) = payload
        .get("survival_time_knots")
        .and_then(serde_json::Value::as_array)
    {
        let max_log_knot = knots
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .filter(|value| value.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        if max_log_knot.is_finite() {
            let hi = max_log_knot.exp();
            if hi.is_finite() && hi > 0.0 {
                return Some(hi);
            }
        }
    }

    // No time-basis knots (the linear Weibull baseline). Two candidate anchors:
    //
    //   * the model's recorded TRAINING time support — the upper end of the
    //     survival exit column's training range — which is the true observed
    //     upper bound of the fitted distribution; and
    //   * a margin over the parametric Weibull `survival_baseline_scale`, to
    //     reach into the right tail.
    //
    // Take the LARGER so the surface grid always covers the observed time range.
    // The scale field alone is unreliable here: on the covariate-driven Weibull
    // parameterization the baseline scale is absorbed into the linear predictor
    // and `survival_baseline_scale` is left as a degenerate floor sentinel
    // (≈ SURVIVAL_TIME_FLOOR ≈ 1e-9). Anchoring on it alone collapsed the grid
    // to ~0 and truncated every `survival_at` query past the prediction frame's
    // `exit` placeholder, even for times well inside the fitted range (#896).
    let mut upper = f64::NEG_INFINITY;
    if let Some(training_hi) = saved_survival_training_exit_upper_bound(payload) {
        upper = upper.max(training_hi);
    }
    if let Some(scale) = payload
        .get("survival_baseline_scale")
        .and_then(serde_json::Value::as_f64)
        && scale.is_finite()
        && scale > 0.0
    {
        upper = upper.max(scale * SURVIVAL_DEFAULT_GRID_SCALE_MARGIN);
    }
    (upper.is_finite() && upper > 0.0).then_some(upper)
}

/// Upper end of the survival exit column's recorded training range.
///
/// Used to anchor the default survival-surface grid to the fitted model's
/// observed time support when the parametric baseline carries no usable time
/// signal (the linear Weibull basis: no knots, and a degenerate
/// `survival_baseline_scale` sentinel — #896). Returns `None` when the model
/// carries no training-range metadata or the exit column cannot be located.
fn saved_survival_training_exit_upper_bound(
    payload: &serde_json::Map<String, serde_json::Value>,
) -> Option<f64> {
    let exit_name = payload
        .get("survival_exit")
        .and_then(serde_json::Value::as_str)?;
    let headers = payload
        .get("training_headers")
        .and_then(serde_json::Value::as_array)?;
    let ranges = payload
        .get("training_feature_ranges")
        .and_then(serde_json::Value::as_array)?;
    let idx = headers.iter().position(|h| h.as_str() == Some(exit_name))?;
    let hi = ranges
        .get(idx)
        .and_then(serde_json::Value::as_array)
        .and_then(|range| range.get(1))
        .and_then(serde_json::Value::as_f64)?;
    (hi.is_finite() && hi > 0.0).then_some(hi)
}

/// Multiplier applied to the Weibull baseline scale when no time-basis knots are
/// available, so the default surface grid reaches into the right tail of the
/// fitted distribution rather than stopping at the characteristic time.
const SURVIVAL_DEFAULT_GRID_SCALE_MARGIN: f64 = 5.0;

#[pyfunction(signature = (model_class, formula, headers, rows, model_bytes = None))]
fn default_survival_time_grid(
    model_class: &str,
    formula: &str,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    model_bytes: Option<Vec<u8>>,
) -> PyResult<Option<Vec<f64>>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    default_survival_time_grid_impl(model_class, formula, &rows.dataset, model_bytes.as_deref())
}

fn default_survival_time_grid_impl(
    model_class: &str,
    formula: &str,
    dataset: &EncodedDataset,
    model_bytes: Option<&[u8]>,
) -> PyResult<Option<Vec<f64>>> {
    match model_class {
        "survival"
        | "competing risks survival"
        | "survival marginal-slope"
        | "survival location-scale" => {}
        _ => return Ok(None),
    }

    // Parse the survival response with the canonical formula parser rather
    // than a bespoke regex. The previous regex only matched the three-argument
    // `Surv(entry, exit, event)` form, so models fit with the right-censored
    // shorthand `Surv(time, event)` (entry synthesized as zero per row) fell
    // through to `None`. That collapsed `model.predict` to a degenerate
    // single-column per-row surface, which `cumulative_hazard_at(grid)` then
    // re-evaluated as a flat constant for every requested time (the time basis
    // never got re-evaluated because there was no real grid). Routing through
    // `parse_surv_response` returns `entry_name: None` for the shorthand, and
    // we span the grid from a synthesized zero entry.
    let parsed = parse_formula(formula)
        .map_err(|err| py_value_error(format!("failed to parse survival formula: {err}")))?;
    let Some((entry_name, exit_name, _event_name)) = parse_surv_response(&parsed.response)
        .map_err(|err| py_value_error(format!("failed to parse Surv(...) response: {err}")))?
    else {
        return Ok(None);
    };

    let header_to_index: HashMap<&str, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    // `entry_name == None` is the two-argument shorthand `Surv(time, event)`:
    // every subject enters at time zero, so the grid lower bound is zero and
    // there is no entry column to read per row.
    let entry_idx = match entry_name.as_deref() {
        Some(name) => match header_to_index.get(name).copied() {
            Some(idx) => Some(idx),
            None => {
                return Err(py_value_error(format!(
                    "survival prediction data is missing required time column(s): {name}"
                )));
            }
        },
        None => None,
    };
    let exit_idx = match header_to_index.get(exit_name.as_str()).copied() {
        Some(idx) => idx,
        None => {
            return Err(py_value_error(format!(
                "survival prediction data is missing required time column(s): {exit_name}"
            )));
        }
    };

    if let Some(index) = entry_idx
        && matches!(
            dataset.schema.columns[index].kind,
            ColumnKindTag::Categorical
        )
    {
        return Err(py_value_error(format!(
            "survival entry column {} is categorical, expected numeric times",
            python_string_repr(entry_name.as_deref().unwrap_or_default())
        )));
    }
    if matches!(
        dataset.schema.columns[exit_idx].kind,
        ColumnKindTag::Categorical
    ) {
        return Err(py_value_error(format!(
            "survival exit column {} is categorical, expected numeric times",
            python_string_repr(&exit_name)
        )));
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for row_index in 0..dataset.values.nrows() {
        let entry_value = match entry_idx {
            None => 0.0,
            Some(index) => dataset.values[[row_index, index]],
        };
        let exit_value = dataset.values[[row_index, exit_idx]];
        if !entry_value.is_finite() || !exit_value.is_finite() {
            return Err(py_value_error(
                "survival time columns must contain only finite values".to_string(),
            ));
        }
        lo = lo.min(entry_value);
        hi = hi.max(exit_value);
    }
    if dataset.values.nrows() == 0 {
        return Ok(None);
    }
    // Anchor the grid's upper edge to the fitted model's training time support,
    // independent of the prediction-frame `exit` placeholder. The placeholder is
    // a semantically meaningless response value that `survival_at` ignores (it
    // supplies its own query times), so it must neither truncate the surface
    // below the fitted range (a SMALL placeholder — issue #896) nor stretch the
    // fixed 64-point uniform grid past the fitted range (a LARGE placeholder,
    // which coarsens every in-range cell and drifts interpolated `survival_at`
    // values — issue #1717). When the training time support is known, it CAPS
    // (and floors) `hi` to that bound regardless of the placeholder; query times
    // legitimately beyond the support are handled by `survival_at`'s
    // extrapolation (#1595), not by the grid. When the model carries no
    // training-time signal (e.g. legacy models) the prediction-frame range is
    // used alone, exactly as before.
    if let Some(bytes) = model_bytes
        && let Some(training_hi) = saved_survival_training_time_upper_bound(bytes)
        && training_hi.is_finite()
    {
        hi = training_hi;
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
    rows: PyRef<'_, PyEncodedTable>,
    formula: String,
    config_json: Option<String>,
    fisher_rao_w: Option<PyReadonlyArray3<'_, f64>>,
) -> PyResult<Py<PyBytes>> {
    // PyO3 0.28 names the old `allow_threads` API `detach`: the closure
    // runs without the GIL, so Python signal handling (KeyboardInterrupt,
    // SIGALRM handlers, etc.) can run while the Rust solver is in progress.
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let fisher_values = fisher_rao_w.as_ref().map(|w| w.as_array().to_owned());
    let model_bytes = detach_workflow_result(py, "fit_table", move || {
        fit_dataset_impl(
            dataset,
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
    let model_bytes = detach_workflow_result(py, "fit_array", move || {
        let dataset = dataset_from_xy_arrays(x_values.view(), y_values.view(), &formula)?;
        fit_dataset_impl(
            dataset,
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
    let payload_a = summary_payload_from_model_bytes(&model_a_bytes)?;
    let payload_b = summary_payload_from_model_bytes(&model_b_bytes)?;
    // Rank on the SAME Occam-penalised conditional AIC that `compare_models`
    // uses (`-2·loglik + 2·edf`, `ranking_score_from_summary_payload`), not the
    // raw REML/LAML evidence headline. The raw headline fails to penalise a
    // pure-noise smooth, so `bayes_factor_vs` used to declare the augmented
    // model better supported even though `compare_models` correctly picked the
    // smaller one — the two contradicted each other (issue #2079).
    let score_a = ranking_score_from_summary_payload(&payload_a)?;
    let score_b = ranking_score_from_summary_payload(&payload_b)?;
    // The ranking score is a minimised cost (lower = better), so the log Bayes
    // factor of A over B is `score_b - score_a`, not `score_a - score_b`. Route
    // through the shared convention so this agrees with `compare_reml_fits`
    // (issue #575: the raw subtraction was inverted, reporting overwhelming
    // evidence for the worse-fitting model).
    //
    // The ranking score is the conditional AIC (`−2·loglik + 2·edf`), a −2·log /
    // deviance-scale cost, so its gap is a ΔAIC. The Akaike evidence ratio for an
    // AIC gap Δ is `exp(−½Δ)` (Burnham & Anderson), so the LOG Bayes factor is
    // HALF the raw score gap. `Model.bayes_factor_vs` exponentiates this value
    // directly; returning the un-halved gap made it report `exp(ΔAIC)`, the SQUARE
    // of the intended ratio (issue #2124). Halve here, at the AIC-scale site, so
    // no raw-REML consumer is affected.
    Ok(0.5 * log_bayes_factor(score_a, score_b))
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

/// Human-readable inference advisories recorded while the model was fit — the
/// mgcv-style "k reduced to the data support" / basis-degradation notes from the
/// cr/cs/sz cap (#1541, #1542), and any other materialization advisory. The CLI
/// prints these; this accessor lets gamfit surface the SAME notes as
/// `GamInferenceWarning`s and via `model.notes` rather than dropping them at the
/// FFI boundary (#1543). Returns an empty list for older payloads that predate
/// the field (it deserializes via `#[serde(default)]`).
#[pyfunction]
fn inference_notes_from_model(model_bytes: Vec<u8>) -> PyResult<Vec<String>> {
    let saved: serde_json::Value = serde_json::from_slice(&model_bytes)
        .map_err(|err| PyValueError::new_err(format!("saved model payload must be JSON: {err}")))?;
    let notes = saved
        .get("payload")
        .and_then(|payload| payload.get("inference_notes"))
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_owned))
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    Ok(notes)
}

fn required_saved_model_payload_string_value(model_bytes: &[u8], key: &str) -> PyResult<String> {
    saved_model_payload_string(model_bytes.to_vec(), key)?
        .ok_or_else(|| py_value_error(format!("saved model payload is missing {key}")))
}

#[pyfunction]
fn required_saved_model_payload_string(model_bytes: Vec<u8>, key: &str) -> PyResult<String> {
    required_saved_model_payload_string_value(&model_bytes, key)
}

/// The canonical fine-grained prediction class label for a saved model — e.g.
/// `"bernoulli marginal-slope"`, `"survival marginal-slope"`, `"competing
/// risks survival"`, `"latent survival"`, `"gaussian location-scale"`,
/// `"transformation-normal"`, or `"standard"`.
///
/// The persisted `model_kind` field is the *coarse* [`ModelKind`] enum, which
/// collapses distinct model classes onto a single tag: both bernoulli- and
/// survival-marginal-slope serialize as `"marginal-slope"`, and every
/// location-scale variant as `"location-scale"`. `gamfit`'s model-introspection
/// sets (`is_marginal_slope`, `is_survival`, `is_transformation_normal`) are
/// written against the fine-grained label, so reading the coarse `model_kind`
/// for them silently misclassifies — e.g. a marginal-slope model reports
/// `is_marginal_slope == False`. This accessor derives the label from the
/// fitted family state, the same authority the predict payloads and CLI use.
#[pyfunction]
fn saved_model_predict_class_name(model_bytes: Vec<u8>) -> PyResult<String> {
    let model: FittedModel = serde_json::from_slice(&model_bytes)
        .map_err(|err| py_value_error(format!("saved model payload must be JSON: {err}")))?;
    Ok(prediction_model_class_label(&model))
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

/// Rewrite smooth-term calls in `formula` so each named smooth carries a
/// `shape=<kind>` DSL option, given a `constraints` mapping serialized as a list
/// of `(term_text, kind)` pairs. All alias normalization, smooth-term scanning,
/// and paren-matching live in
/// [`gam::terms::smooth::apply_shape_constraints_to_formula`]; the Python
/// wrapper only marshals the dict across the FFI.
#[pyfunction]
fn apply_shape_constraints_to_formula(
    formula: String,
    constraints: Vec<(String, String)>,
) -> PyResult<String> {
    gam::terms::smooth::apply_shape_constraints_to_formula(&formula, &constraints)
        .map_err(py_value_error)
}

#[pyfunction]
fn validate_formula_json(
    py: Python<'_>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    formula: String,
    config_json: Option<String>,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    detach_py_result(py, "validate_formula_json", move || {
        validate_formula_dataset_json_impl(dataset, formula, config_json.as_deref())
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
        rows.push_str("<th style='text-align:left;padding:0.25rem 0.75rem 0.25rem 0;'>");
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

#[pyfunction(signature = (interval, time_grid, covariance_mode=None, observation_interval=None))]
fn build_predict_payload_json(
    interval: Option<f64>,
    time_grid: Option<Vec<f64>>,
    covariance_mode: Option<String>,
    observation_interval: Option<bool>,
) -> PyResult<String> {
    // Validate the covariance-mode string here (before transport) so a typo
    // surfaces as a clear error at the predict call site rather than as an
    // opaque deserialization failure later.
    parse_covariance_mode(covariance_mode.as_deref()).map_err(py_value_error)?;
    let payload = PyPredictOptionsPayload {
        interval,
        time_grid,
        covariance_mode,
        observation_interval,
        conformal_level: None,
    };
    serde_json::to_string(&payload)
        .map_err(|err| py_value_error(format!("failed to serialize predict payload: {err}")))
}

#[pyfunction(signature = (model_bytes, headers, rows, interval, covariance_mode=None, observation_interval=None))]
fn build_model_predict_payload_json(
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    interval: Option<f64>,
    covariance_mode: Option<String>,
    observation_interval: Option<bool>,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let model_class = required_saved_model_payload_string_value(&model_bytes, "model_kind")?;
    let formula = required_saved_model_payload_string_value(&model_bytes, "formula")?;
    let time_grid =
        default_survival_time_grid_impl(&model_class, &formula, &rows.dataset, Some(&model_bytes))?;
    build_predict_payload_json(interval, time_grid, covariance_mode, observation_interval)
}

#[pyfunction]
fn predict_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    options_json: Option<String>,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    detach_predict_result(py, "predict_table", move || {
        predict_encoded_table_impl(&model_bytes, dataset, options_json.as_deref())
    })
}

/// Distribution-free conformal prediction intervals (issue #310 family path).
///
/// Runs the standard model-based predictor on `(headers, rows)`, then replaces
/// the response-scale `mean_lower` / `mean_upper` with the split-conformal
/// interval `μ̂(x) ± q̂·s(x)` calibrated at `conformal_level` from the held-out
/// `(calibration_headers, calibration_rows)` fold — which must contain the
/// response column. The returned interval carries finite-sample marginal
/// coverage `≥ conformal_level` regardless of model misspecification.
#[pyfunction(signature = (model_bytes, headers, rows, calibration_headers, calibration_rows, conformal_level, options_json=None))]
fn predict_table_conformal(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    calibration_headers: Vec<String>,
    calibration_rows: PyRef<'_, PyEncodedTable>,
    conformal_level: f64,
    options_json: Option<String>,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    calibration_rows
        .require_headers(&calibration_headers)
        .map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let calibration_dataset = calibration_rows.dataset.clone();
    detach_py_result(py, "predict_table_conformal", move || {
        predict_encoded_table_conformal_impl(
            &model_bytes,
            dataset,
            calibration_dataset,
            conformal_level,
            options_json.as_deref(),
        )
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
    let (cif, overall_survival) = detach_pyresult(py, "competing_risks_cif", move || {
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
) -> PyResult<(Array3<f64>, Array2<f64>)> {
    let endpoint_views = cumulative_hazards
        .iter()
        .map(|hazard| hazard.view())
        .collect::<Vec<_>>();
    // `ndarray::stack` is a pure shape contract violation — keep it as a
    // bare `PyValueError` rather than forcing it through a typed engine
    // enum it does not belong to.
    let cumulative_hazard =
        ndarray::stack(Axis(0), &endpoint_views).map_err(shape_error_to_pyerr)?;
    // Typed engine path: `assemble_competing_risks_cif` returns
    // `Result<_, SurvivalError>`, dispatch to `gamfit.SurvivalError`.
    let result =
        gam::families::survival::assemble_competing_risks_cif(times, cumulative_hazard.view())
            .map_err(survival_error_to_pyerr)?;
    let cif_views = result.cif.iter().map(|m| m.view()).collect::<Vec<_>>();
    let cif_stacked = ndarray::stack(Axis(0), &cif_views).map_err(shape_error_to_pyerr)?;
    Ok((cif_stacked, result.overall_survival))
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
                "all endpoint predictions must return the same (n_rows, n_times) shape".to_string(),
            ));
        }
    }

    let cumulative_hazard_values = cumulative_hazards
        .iter()
        .map(|hazard| hazard.as_array().to_owned())
        .collect::<Vec<_>>();
    let (cif, overall_survival) =
        detach_pyresult(py, "competing_risks_cif_from_predictions", move || {
            competing_risks_cif_from_predictions_impl(time_values.view(), &cumulative_hazard_values)
        })?;
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
) -> PyResult<(Vec<Array2<f64>>, Array2<f64>)> {
    // Typed engine path: `SurvivalError` → `gamfit.SurvivalError` (issue
    // #343), no string flattening.
    let result = gam::families::survival::assemble_competing_risks_cif_from_endpoints(
        times,
        cumulative_hazards,
    )
    .map_err(survival_error_to_pyerr)?;
    let cif = result.cif;
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
    serde_json::to_string(&serde_json::Value::Object(payload))
        .map_err(|err| py_value_error(format!("failed to serialize sample options json: {err}")))
}

#[pyfunction]
fn sample_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    options_json: Option<String>,
) -> PyResult<Py<PyDict>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let payload = detach_py_result(py, "sample_table", move || {
        sample_encoded_table_impl(&model_bytes, dataset, options_json.as_deref())
    })?;
    let config = PyDict::new(py);
    config.set_item("n_samples", payload.config.n_samples)?;
    config.set_item("n_warmup", payload.config.n_warmup)?;
    config.set_item("n_chains", payload.config.n_chains)?;
    config.set_item("target_accept", payload.config.target_accept)?;
    config.set_item("seed", payload.config.seed)?;
    let out = PyDict::new(py);
    out.set_item("samples", payload.samples.into_pyarray(py))?;
    out.set_item("coefficient_names", payload.coefficient_names)?;
    out.set_item("posterior_mean", payload.posterior_mean.into_pyarray(py))?;
    out.set_item("posterior_std", payload.posterior_std.into_pyarray(py))?;
    out.set_item("rhat", payload.rhat)?;
    out.set_item("ess", payload.ess)?;
    out.set_item("converged", payload.converged)?;
    out.set_item("config", config)?;
    out.set_item("model_class", payload.model_class)?;
    out.set_item("family_kind", payload.family_kind)?;
    out.set_item("link_spec", payload.link_spec)?;
    out.set_item("method", payload.method)?;
    Ok(out.unbind())
}

// paired_sample_table and paired_cumulative_incidence_table pyffi wrappers
// removed: their payload types (PairedSamplePayload, PairedCifPayload,
// PyPairedCifOptions) were never defined in the main crate; the orphaned
// implementations and their helpers were deleted below.

#[pyfunction]
fn design_matrix_table_dense(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
) -> PyResult<Py<PyArray2<f64>>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let out = detach_py_result(py, "design_matrix_table_dense", move || {
        design_matrix_encoded_table_impl(&model_bytes, dataset)
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

/// Build a closed cyclic uniform B-spline basis and its exact derivative
/// roughness penalty on the periodic parameter `t`.
///
/// The basis lives on `[0, 1)` (values of `t` are reduced modulo 1 by the
/// underlying kernel) with `n_knots` cyclic control points, and the
/// returned penalty is `∮(f^(penalty_order))²` in that basis (the constant
/// function is its only nullspace direction).
#[pyfunction(signature = (t, n_knots, degree = 3, penalty_order = 2))]
fn periodic_spline_curve_basis<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    n_knots: usize,
    degree: usize,
    penalty_order: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let spec = PeriodicBSplineBasisSpec::new(degree, n_knots, 1.0, 0.0, penalty_order);
    let basis =
        build_periodic_bspline_basis_1d(t.as_array(), &spec).map_err(basis_error_to_pyerr)?;
    let penalty = cyclic_bspline_derivative_penalty_matrix(degree, n_knots, 1.0, penalty_order)
        .map_err(basis_error_to_pyerr)?;
    Ok((
        basis.into_pyarray(py).unbind(),
        penalty.into_pyarray(py).unbind(),
    ))
}

/// Exact periodic B-spline function roughness
/// `S_ab = ∮ B_a^(order)(t) B_b^(order)(t) dt` over one period.
#[pyfunction(signature = (num_basis, degree = 3, period = 1.0, order = 2))]
fn cyclic_bspline_roughness_penalty(
    py: Python<'_>,
    num_basis: usize,
    degree: usize,
    period: f64,
    order: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let penalty = cyclic_bspline_derivative_penalty_matrix(degree, num_basis, period, order)
        .map_err(basis_error_to_pyerr)?;
    Ok(penalty.into_pyarray(py).unbind())
}

/// Exact continuum shape cone `A * beta >= b` for raw open-B-spline control
/// coefficients. The Rust terms layer owns the knot geometry so Torch and the
/// native fit path cannot drift into different monotonicity/curvature rules.
#[pyfunction]
fn bspline_shape_constraints(
    py: Python<'_>,
    knots: PyReadonlyArray1<'_, f64>,
    degree: usize,
    shape: &str,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let shape = gam::terms::smooth::parse_shape_constraint(shape).map_err(py_value_error)?;
    let constraints = gam::terms::smooth::shape_constraints::bspline_shape_linear_constraints(
        knots.as_array(),
        degree,
        shape,
    )
    .map_err(basis_error_to_pyerr)?
    .ok_or_else(|| {
        py_value_error("bspline_shape_constraints requires a non-None shape".to_string())
    })?;
    Ok((
        constraints.a.into_pyarray(py).unbind(),
        constraints.b.into_pyarray(py).unbind(),
    ))
}

fn build_wrapped_periodic_harmonic_basis_with_jet(
    t: ArrayView1<'_, f64>,
    n_harmonics: usize,
    label: &str,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    if t.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} requires finite t values"));
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
            let angle = frequency * t[row].rem_euclid(1.0);
            let sin_value = angle.sin();
            let cos_value = angle.cos();

            phi[[row, sin_col]] = sin_value;
            phi[[row, cos_col]] = cos_value;
            jet[[row, sin_col, 0]] = frequency * cos_value;
            jet[[row, cos_col, 0]] = -frequency * sin_value;
        }
    }

    Ok((phi, jet, penalty))
}

#[pyfunction]
fn periodic_basis_with_jet<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    n_harmonics: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>, Py<PyArray2<f64>>)> {
    let (phi, jet, penalty) = build_wrapped_periodic_harmonic_basis_with_jet(
        t.as_array(),
        n_harmonics,
        "periodic_basis_with_jet",
    )
    .map_err(py_value_error)?;

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
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>, Py<PyArray2<f64>>)> {
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
        radial_reparam: None,
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
    // The pure scale-free Duchon design (`power = 0`, `length_scale = None`)
    // and its analytic input-location first jet are built *together* by the
    // single Rust-core helper `duchon_sae_atom_basis_with_jet`. Building both
    // from one constraint null space `Z` and one kernel amplification `α`
    // guarantees the returned jet is the exact `t`-derivative of the returned
    // `Φ` column-for-column (`J_kernel = α·K'(t,C)·Z`, polynomial columns
    // carrying their own monomial derivative and no amplification) — there is
    // no second, independently-scaled forward pipeline that could drift out of
    // lockstep on the amplification, the null-space basis, or the polynomial
    // column ordering. This is the model the issue prescribes.
    let (phi, jet) = duchon_sae_atom_basis_with_jet(pts, ctrs, requested_nullspace)
        .map_err(basis_error_to_pyerr)?;

    // The penalty matrix `S = Zᵀ K_CC Z` is the conditionally-PD penalty of the
    // *same* basis. It comes from the forward builder over the identical spec,
    // which uses the same `Z` and `α` as the helper above, so `S` is the
    // penalty of exactly the `Φ` returned here.
    let built = build_duchon_basis(pts, &spec).map_err(basis_error_to_pyerr)?;
    let primary_idx = built
        .penaltyinfo
        .iter()
        .position(|info| matches!(info.source, gam::terms::basis::PenaltySource::Primary))
        .ok_or_else(|| {
            py_value_error("duchon_basis_with_jet: primary penalty was not built".to_string())
        })?;
    let penalty = built.penalties[primary_idx].clone();

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

/// Forward Duchon design **and** its analytic input-location first and second
/// jets, all built from the *same* resolved spec the `duchon_basis` forward
/// uses. This is the autograd companion the Python `Duchon` descriptor needs:
/// the returned jets are the exact derivatives of the built design
/// `X(x) = [α·K(x,C)·Z, P(x)]` — including the polynomial-constraint null-space
/// projection `Z`, the appended polynomial nullspace columns `P(x)`, the kernel
/// amplification `α`, the hybrid length-scale / power spectrum, and the
/// periodic chord embedding — not the raw centerwise radial kernel.
///
/// Returns `(Φ, J, H)` with `Φ` shape `(N, M)`, `J` shape `(N, M, d)`, and
/// `H` shape `(N, M, d, d)`, where `M = n_kernel + n_poly`. The keyword
/// surface and resolution policy are identical to [`duchon_basis`] so the
/// forward block of `Φ` is bit-equal to a standalone `duchon_basis` call.
#[pyfunction(signature = (
    points,
    centers,
    m = 2,
    periodic_per_axis = None,
    length_scale = None,
    nullspace_order = "linear",
    power = None,
))]
fn duchon_basis_with_jets<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    m: usize,
    periodic_per_axis: Option<Vec<bool>>,
    length_scale: Option<f64>,
    nullspace_order: Option<&str>,
    power: Option<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>, Py<PyArray4<f64>>)> {
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
            "duchon_basis_with_jets requires finite points and centers".to_string(),
        ));
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

    // Resolve the (nullspace_order, power) pair exactly as the basis-only
    // `duchon_basis` forward does (`max_op = 0`): the returned jets must
    // differentiate the *same* matrix the forward builds.
    let any_periodic = periodic_flags.iter().any(|&b| b);
    let cfg = resolve_duchon_hybrid_config(
        d,
        length_scale,
        nullspace_order,
        power,
        /* max_op = */ 0,
        any_periodic,
    )?;

    // Periods for periodic axes: auto-derived as (max − min) over centers,
    // matching `build_duchon_basis_mixed_periodicity_auto`'s `periods = None`
    // policy; non-periodic axes carry an unused placeholder.
    let mut periods = vec![1.0_f64; d];
    for (j, &per) in periodic_flags.iter().enumerate() {
        if per {
            let col = ctrs.column(j);
            let left = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let right = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if !left.is_finite() || !right.is_finite() || left >= right {
                return Err(py_value_error(format!(
                    "duchon_basis_with_jets: periodic axis {j} has degenerate center span [{left}, {right}]"
                )));
            }
            periods[j] = right - left;
        }
    }

    let (phi, jet, hess) = gam::terms::basis::build_duchon_basis_design_and_jets(
        pts,
        ctrs,
        cfg.length_scale,
        cfg.power,
        cfg.nullspace_order,
        &periodic_flags,
        &periods,
    )
    .map_err(basis_error_to_pyerr)?;

    Ok((
        phi.into_pyarray(py).unbind(),
        jet.into_pyarray(py).unbind(),
        hess.into_pyarray(py).unbind(),
    ))
}

/// Evaluate the Matérn kernel basis design matrix at `points` against `centers`.
///
/// `points` is `(N, d)`, `centers` is `(K, d)`. `nu` accepted as `"1/2"`,
/// `"3/2"`, `"5/2"`, `"7/2"`, or `"9/2"` (also accepts `"0.5"`/etc). Forward
/// only; gradients with respect to `points` are exposed via
/// `matern_input_location_first_jet` (the full `∂Φ/∂t` tensor under the same
/// anisotropic metric) and `matern_input_location_hessian` (the matching
/// second derivative).
#[pyfunction(signature = (points, centers, length_scale = 1.0, nu = "3/2", aniso_log_scales = None))]
fn matern_basis<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    length_scale: f64,
    nu: &str,
    aniso_log_scales: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let pts = points.as_array();
    let ctrs = centers.as_array();
    if pts.ncols() != ctrs.ncols() {
        return Err(py_value_error(format!(
            "matern_basis: points has d={} but centers has d={}",
            pts.ncols(),
            ctrs.ncols()
        )));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(py_value_error(format!(
            "matern_basis: length_scale must be finite and > 0, got {length_scale}"
        )));
    }
    if pts.iter().any(|value| !value.is_finite()) || ctrs.iter().any(|value| !value.is_finite()) {
        return Err(py_value_error(
            "matern_basis: points and centers must be finite".to_string(),
        ));
    }
    let nu_parsed = parse_matern_nu_py("matern_basis", nu)?;
    let aniso_vec = aniso_log_scales
        .as_ref()
        .map(|values| values.as_slice())
        .transpose()
        .map_err(|err| py_value_error(format!("aniso_log_scales must be contiguous: {err}")))?
        .map(|slice| slice.to_vec());
    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
        length_scale,
        nu: nu_parsed,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::None,
        aniso_log_scales: aniso_vec,
        periodic: None,
        nullspace_shrinkage_survived: None,
    };
    // Honor an explicit all-zero `aniso_log_scales` literally as the isotropic
    // metric — this is a caller's explicit request, NOT the κ-optimizer's
    // geometry-seeding sentinel (#1042).
    let built = build_matern_basis_literal_aniso(pts, &spec).map_err(basis_error_to_pyerr)?;
    let design = built
        .design
        .try_to_dense_by_chunks("matern_basis")
        .map_err(py_value_error)?;
    Ok(design.into_pyarray(py).unbind())
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
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>, Py<PyArray2<f64>>)> {
    match kind.to_ascii_lowercase().replace('-', "_").as_str() {
        "duchon" | "euclidean" | "euclidean_patch" => {
            let centers = params
                .get_item("centers")?
                .ok_or_else(|| {
                    py_value_error("basis_with_jet params missing \"centers\"".to_string())
                })?
                .extract::<PyReadonlyArray2<'py, f64>>()?;
            let m = required_usize_param(params, "m")?;
            duchon_basis_with_jet(py, t, centers, m)
        }
        "periodic" | "periodic_spline" | "circle" => {
            let n_harmonics = required_usize_param(params, "n_harmonics")?;
            let coords = t.as_array();
            if coords.ncols() != 1 {
                return Err(py_value_error(format!(
                    "basis_with_jet periodic basis is intrinsically 1D and requires t with exactly one column, got shape ({}, {})",
                    coords.nrows(),
                    coords.ncols()
                )));
            }
            let (phi, jet, penalty) = build_wrapped_periodic_harmonic_basis_with_jet(
                coords.column(0),
                n_harmonics,
                "basis_with_jet periodic basis",
            )
            .map_err(py_value_error)?;

            Ok((
                phi.into_pyarray(py).unbind(),
                jet.into_pyarray(py).unbind(),
                penalty.into_pyarray(py).unbind(),
            ))
        }
        "sphere" => sphere_chart_basis_with_jet(py, t),
        "bspline" | "b_spline" | "b-spline" => {
            let coords = t.as_array();
            if coords.ncols() != 1 {
                return Err(py_value_error(format!(
                    "basis_with_jet bspline basis is intrinsically 1D and requires t with exactly one column, got shape ({}, {})",
                    coords.nrows(),
                    coords.ncols()
                )));
            }
            if coords.iter().any(|value| !value.is_finite()) {
                return Err(py_value_error(
                    "basis_with_jet bspline basis requires finite t values".to_string(),
                ));
            }
            let degree = params
                .get_item("degree")?
                .map(|v| v.extract::<usize>())
                .transpose()?
                .unwrap_or(3);
            let order = params
                .get_item("order")?
                .map(|v| v.extract::<usize>())
                .transpose()?
                .unwrap_or(2);
            let periodic = params
                .get_item("periodic")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(false);
            let knots_array: Array1<f64> = match params.get_item("knots")? {
                Some(obj) => obj
                    .extract::<PyReadonlyArray1<'py, f64>>()?
                    .as_array()
                    .to_owned(),
                None => {
                    let n_basis = params
                        .get_item("n_basis")?
                        .map(|v| v.extract::<usize>())
                        .transpose()?
                        .ok_or_else(|| {
                            py_value_error(
                                "basis_with_jet bspline params require either \"knots\" or \"n_basis\""
                                    .to_string(),
                            )
                        })?;
                    if n_basis < degree + 1 {
                        return Err(py_value_error(format!(
                            "basis_with_jet bspline: n_basis ({n_basis}) must be >= degree+1 ({})",
                            degree + 1
                        )));
                    }
                    let interior = n_basis.saturating_sub(degree + 1);
                    let total = interior + 2 * (degree + 1);
                    let mut knots = Array1::<f64>::zeros(total);
                    let inner = interior as f64 + 1.0;
                    for i in 0..total {
                        let raw = (i as f64) - (degree as f64);
                        let clamped = raw.max(0.0).min(inner);
                        knots[i] = clamped / inner;
                    }
                    knots
                }
            };
            let t_1d = coords.column(0).to_owned();
            let phi = bspline_basis_impl(t_1d.view(), knots_array.view(), degree, periodic)
                .map_err(py_value_error)?;
            let n_rows = phi.nrows();
            let n_cols = phi.ncols();
            // The forward periodic basis (`bspline_basis_impl(..., periodic=true)`)
            // is the normalized cyclic B-spline on the closed parameter circle
            // `[left, right]` with `num_basis = knots.len() - 1` cyclic control
            // points (see `periodic_knot_domain`). Its exact input-location
            // derivative is the periodic wrapped-spline jet — the SAME pair the
            // PyTorch `_BsplineJetFn` uses — NOT the Fourier harmonic basis
            // produced by `kind="periodic"`. So differentiate the actual design
            // matrix here via `periodic_bspline_first_derivative_nd`, which is
            // the closed form `phi'` that `PeriodicSplineCurve::evaluate_derivative`
            // also relies on. The non-periodic branch uses the open-uniform
            // analytic first derivative.
            let (jet, penalty) = if periodic {
                let (left, right, num_basis) =
                    periodic_knot_domain(knots_array.view()).map_err(py_value_error)?;
                let jet =
                    periodic_bspline_first_derivative_nd(coords, (left, right), degree, num_basis)
                        .map_err(basis_error_to_pyerr)?;
                if jet.shape() != [n_rows, n_cols, 1] {
                    return Err(py_value_error(format!(
                        "basis_with_jet bspline shape mismatch: phi=({n_rows},{n_cols}) jet={:?}",
                        jet.shape()
                    )));
                }
                let penalty = cyclic_bspline_derivative_penalty_matrix(
                    degree,
                    num_basis,
                    right - left,
                    order,
                )
                .map_err(basis_error_to_pyerr)?;
                (jet, penalty)
            } else {
                let deriv = bspline_basis_derivative_impl(
                    t_1d.view(),
                    knots_array.view(),
                    degree,
                    1,
                    false,
                )
                .map_err(py_value_error)?;
                if deriv.nrows() != n_rows || deriv.ncols() != n_cols {
                    return Err(py_value_error(format!(
                        "basis_with_jet bspline shape mismatch: phi=({n_rows},{n_cols}) deriv=({},{})",
                        deriv.nrows(),
                        deriv.ncols()
                    )));
                }
                let mut jet = Array3::<f64>::zeros((n_rows, n_cols, 1));
                for row in 0..n_rows {
                    for col in 0..n_cols {
                        jet[[row, col, 0]] = deriv[[row, col]];
                    }
                }
                let (penalty, null_basis) =
                    smoothness_penalty_impl(knots_array.view(), degree, order)
                        .map_err(py_value_error)?;
                assert!(
                    null_basis.ncols() <= penalty.ncols(),
                    "smoothness penalty nullspace cannot exceed coefficient count"
                );
                (jet, penalty)
            };
            Ok((
                phi.into_pyarray(py).unbind(),
                jet.into_pyarray(py).unbind(),
                penalty.into_pyarray(py).unbind(),
            ))
        }
        "linear" | "affine" | "linear_rank1" => {
            // A genuinely linear (rank-1 affine) atom carries the curve γ(t)=t·b,
            // so each coordinate axis IS its own basis function: φ(t)=t with a
            // constant identity input-derivative and no curvature to penalize.
            // For the scalar SAE projection (d=1) this is the single-column design
            // φ=t, jet ∂φ/∂t=1, penalty=[0] — exactly the M_k=1 linear comparison
            // arm the #1026 EV-vs-K ladder fits against the curved atom.
            let coords = t.as_array();
            if coords.iter().any(|value| !value.is_finite()) {
                return Err(py_value_error(
                    "basis_with_jet linear basis requires finite t values".to_string(),
                ));
            }
            let n_rows = coords.nrows();
            let d = coords.ncols();
            if d == 0 {
                return Err(py_value_error(
                    "basis_with_jet linear basis requires t with at least one column".to_string(),
                ));
            }
            let phi = coords.to_owned();
            let mut jet = Array3::<f64>::zeros((n_rows, d, d));
            for row in 0..n_rows {
                for axis in 0..d {
                    jet[[row, axis, axis]] = 1.0;
                }
            }
            // Linear functions lie entirely in the smoothing nullspace: no
            // second-derivative energy, hence a zero curvature penalty.
            let penalty = Array2::<f64>::zeros((d, d));
            Ok((
                phi.into_pyarray(py).unbind(),
                jet.into_pyarray(py).unbind(),
                penalty.into_pyarray(py).unbind(),
            ))
        }
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
    nullspace_order = "linear",
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
    // Basis-only PyFFI: the returned object is just the (N, K) design
    // matrix; no operator penalty crosses the FFI. The downstream basis
    // builder needs only kernel-existence (`2(p+s) > d`) and — in the
    // pure-Duchon case — the CPD/nullspace adequacy guard (`2s < d`).
    // D1 / D2 collocation are *not* required here, so resolve with
    // ``max_op = 0`` and construct the spec with all three operator
    // penalties Disabled. This makes documented defaults (e.g. d=2 m=2
    // thin-plate, d=3 m=2 generalized TPS) succeed without forcing the
    // caller to pass ``power`` themselves.
    let cfg = resolve_duchon_hybrid_config(
        d,
        length_scale,
        nullspace_order,
        power,
        /* max_op = */ 0,
        any_periodic,
    )?;
    let (spec_length_scale, spec_nullspace, spec_power) =
        (cfg.length_scale, cfg.nullspace_order, cfg.power);
    let basis_only_operator_penalties = DuchonOperatorPenaltySpec {
        mass: OperatorPenaltySpec::Disabled,
        tension: OperatorPenaltySpec::Disabled,
        stiffness: OperatorPenaltySpec::Disabled,
    };
    // Any periodic axis (1D or multi-D) routes through the mixed-periodicity
    // builder (cylinder/torus chord-distance polyharmonic).
    if any_periodic {
        let spec = DuchonBasisSpec {
            radial_reparam: None,
            center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
            length_scale: spec_length_scale,
            power: spec_power,
            nullspace_order: spec_nullspace,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: basis_only_operator_penalties.clone(),
            periodic: None,
            boundary: OneDimensionalBoundary::Open,
        };
        let built = build_duchon_basis_mixed_periodicity_auto(pts, &spec, &periodic_flags, None)
            .map_err(basis_error_to_pyerr)?;
        return Ok(built.design.to_dense().into_pyarray(py).unbind());
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
        length_scale: spec_length_scale,
        power: spec_power,
        nullspace_order: spec_nullspace,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: basis_only_operator_penalties,
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    let built = build_duchon_basis(pts, &spec).map_err(basis_error_to_pyerr)?;
    Ok(built.design.to_dense().into_pyarray(py).unbind())
}

#[pyfunction(signature = (t, num_internal_knots, degree = 3))]
fn auto_knots_1d<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    num_internal_knots: usize,
    degree: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize, usize, bool)> {
    // Issue #340: return the auto-shrunk effective `(degree, num_internal_knots)`
    // alongside the knot vector so Python callers can observe when the engine
    // had to downgrade their requested basis to fit small-n data.
    let result = auto_knot_vector_1d_quantile(t.as_array(), num_internal_knots, degree)
        .map_err(basis_error_to_pyerr)?;
    Ok((
        result.knots.into_pyarray(py).unbind(),
        result.degree,
        result.num_internal_knots,
        result.shrunk,
    ))
}

#[pyfunction(signature = (t, num_centers))]
fn auto_centers_1d<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    num_centers: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let centers =
        auto_centers_1d_equal_mass(t.as_array(), num_centers).map_err(basis_error_to_pyerr)?;
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
    .map_err(basis_error_to_pyerr)?;
    Ok((
        matrices.mass.into_pyarray(py).unbind(),
        matrices.tension.into_pyarray(py).unbind(),
        matrices.stiffness.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (
    centers,
    m = 2,
    period = None,
    periodic_per_axis = None,
    length_scale = None,
    nullspace_order = "linear",
    power = None,
))]
fn duchon_function_norm_penalty<'py>(
    py: Python<'py>,
    centers: PyReadonlyArray2<'py, f64>,
    m: usize,
    period: Option<f64>,
    periodic_per_axis: Option<Vec<bool>>,
    length_scale: Option<f64>,
    nullspace_order: Option<&str>,
    power: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    if m == 0 {
        return Err(py_value_error("Duchon m must be at least 1".to_string()));
    }
    let center_matrix: Array2<f64> = centers.as_array().to_owned();
    let d = center_matrix.ncols();
    let periodic_flags: Vec<bool> = if let Some(flags) = periodic_per_axis.clone() {
        if flags.len() != d {
            return Err(py_value_error(format!(
                "periodic_per_axis must have length d={}, got {}",
                d,
                flags.len()
            )));
        }
        flags
    } else {
        vec![false; d]
    };
    let any_periodic = periodic_flags.iter().any(|&b| b);
    if d == 1 {
        let col = center_matrix.column(0);
        validate_position_period("duchon", col, any_periodic, period).map_err(py_value_error)?;
    } else if period.is_some() {
        return Err(py_value_error(
            "duchon scalar `period` is only valid for d=1 (multi-D periodic axes auto-derive period from centers)".to_string(),
        ));
    }
    let (spec_length_scale, spec_nullspace, spec_power) = match power {
        Some(explicit_power) => {
            let cfg = resolve_duchon_hybrid_config(
                d,
                length_scale,
                nullspace_order,
                Some(explicit_power),
                /* max_op = */ 0,
                any_periodic,
            )?;
            (cfg.length_scale, cfg.nullspace_order, cfg.power)
        }
        None => {
            let cfg = resolve_duchon_hybrid_config(
                d,
                length_scale,
                nullspace_order,
                None,
                0,
                any_periodic,
            )?;
            (cfg.length_scale, cfg.nullspace_order, cfg.power)
        }
    };
    // Any periodic axis (1D or multi-D) routes through the mixed-periodicity
    // builder (cylinder/torus chord-distance polyharmonic).
    if any_periodic {
        let spec = DuchonBasisSpec {
            radial_reparam: None,
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
        // Honor an explicit 1D `period` (the domain wrap) instead of
        // auto-deriving it from the center span, which undershoots on a
        // half-open grid and produced a non-PSD Gram (gam#580). For d>1 the
        // per-axis periods are auto-derived in the core.
        let periods_1d: Option<[f64; 1]> = if d == 1 { period.map(|p| [p]) } else { None };
        let built = build_duchon_basis_mixed_periodicity_auto(
            center_matrix.view(),
            &spec,
            &periodic_flags,
            periods_1d.as_ref().map(|p| p.as_slice()),
        )
        .map_err(basis_error_to_pyerr)?;
        // Mixed-periodicity builder emits a single Primary candidate (the
        // function-norm Gram).
        let idx = built
            .penaltyinfo
            .iter()
            .position(|info| matches!(info.source, gam::terms::basis::PenaltySource::Primary))
            .ok_or_else(|| {
                py_value_error(
                    "mixed-periodicity Duchon function-norm penalty was not built".to_string(),
                )
            })?;
        let penalty = built.penalties[idx].clone();
        return Ok(penalty.into_pyarray(py).unbind());
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
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
    let built = build_duchon_basis(center_matrix.view(), &spec).map_err(basis_error_to_pyerr)?;
    // The redesigned non-periodic Euclidean path emits a single native
    // reproducing-norm Gram as the `Primary` candidate (the function-norm
    // penalty on the scale-free polyharmonic basis) plus a null-space shrinkage
    // ridge; it no longer ships the mass/tension/stiffness operator triplet.
    // The function norm is the Primary block.
    let primary_idx = built
        .penaltyinfo
        .iter()
        .position(|info| matches!(info.source, gam::terms::basis::PenaltySource::Primary))
        .ok_or_else(|| {
            py_value_error(
                "Duchon function-norm penalty (Primary native-norm Gram) was not built".to_string(),
            )
        })?;
    let penalty = built.penalties[primary_idx].clone();
    Ok(penalty.into_pyarray(py).unbind())
}

/// Build the spherical-spline (S²) basis and matching penalty matrix.
///
/// `points` is an `(N, 2)` array of latitude/longitude pairs (degrees by
/// default, radians when `radians=True`). The role of `n_centers` depends on
/// the kernel:
///
/// * `"sobolev"` — the finite Wahba center kernel; `n_centers` is the number
///   of centers and therefore the basis dimension `K`.
/// * `"harmonic"` — a truncated spherical-harmonic basis of degree
///   `L = n_centers` (basis dim `K = L * (L + 2)`).
/// * `"pseudo"` — the pseudodifferential kernel is resolved by the builder to
///   the harmonic engine (its low-degree start avoids the finite-center
///   constant-collision the Wahba chart hits; see `term_design`). Here
///   `n_centers` is a *target width* that selects a harmonic degree `L`, so
///   the basis dimension is `K = L * (L + 2)`, not the literal `n_centers`.
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
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
    };
    let built = build_spherical_spline_basis(pts, &spec).map_err(basis_error_to_pyerr)?;
    let primary_idx = built
        .penaltyinfo
        .iter()
        .position(|info| matches!(info.source, gam::terms::basis::PenaltySource::Primary))
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

/// Farthest-point center selection on S² (lat/lon).
///
/// Returns the `(n_centers, 2)` matrix of selected centers in the same
/// angular convention (`radians` flag) as the input. Centers are a property
/// of the basis — once selected they are independent of any future
/// evaluation set, so callers should cache the result.
#[pyfunction(signature = (points, n_centers, radians = false))]
fn sphere_select_farthest_point_centers<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    n_centers: usize,
    radians: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let pts = points.as_array();
    if pts.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_select_farthest_point_centers expects points of shape (N, 2); got d={}",
            pts.ncols()
        )));
    }
    let centers = select_spherical_farthest_point_centers(pts, n_centers, radians)
        .map_err(basis_error_to_pyerr)?;
    Ok(centers.into_pyarray(py).unbind())
}

/// Spherical-spline basis evaluated against explicit (caller-supplied) centers.
///
/// Unlike `sphere_basis`, the basis dimension here is fixed by `centers.nrows()`
/// and is independent of `points.nrows()`. This is the correct path whenever
/// the caller has already resolved a center set (e.g. from training data, an
/// explicit user spec, or a deterministic sphere lattice).
///
/// For `kernel = "harmonic"`, centers act only as a degree probe:
/// `max_degree = centers.nrows()`; the actual points are not sampled from
/// the center set.
#[pyfunction(signature = (points, centers, penalty_order = 2, kernel = "sobolev", radians = false))]
fn sphere_basis_with_centers<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    penalty_order: usize,
    kernel: &str,
    radians: bool,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let pts = points.as_array();
    let ctrs = centers.as_array();
    if pts.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_basis_with_centers expects points of shape (N, 2) [lat, lon]; got d={}",
            pts.ncols()
        )));
    }
    if ctrs.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_basis_with_centers expects centers of shape (K, 2) [lat, lon]; got d={}",
            ctrs.ncols()
        )));
    }
    if !(1..=4).contains(&penalty_order) {
        return Err(py_value_error(format!(
            "sphere_basis_with_centers penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        )));
    }
    let (method, wahba_kernel, max_degree) = match kernel.to_ascii_lowercase().as_str() {
        "sobolev" => (SphereMethod::Wahba, SphereWahbaKernel::Sobolev, None),
        "pseudo" => (SphereMethod::Wahba, SphereWahbaKernel::Pseudo, None),
        "harmonic" => (
            SphereMethod::Harmonic,
            SphereWahbaKernel::Sobolev,
            Some(ctrs.nrows()),
        ),
        other => {
            return Err(py_value_error(format!(
                "sphere_basis_with_centers kernel must be one of 'sobolev', 'pseudo', 'harmonic'; got '{other}'"
            )));
        }
    };
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
        penalty_order,
        double_penalty: false,
        radians,
        method,
        max_degree,
        wahba_kernel,
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
    };
    let built = build_spherical_spline_basis(pts, &spec).map_err(basis_error_to_pyerr)?;
    let primary_idx = built
        .penaltyinfo
        .iter()
        .position(|info| matches!(info.source, gam::terms::basis::PenaltySource::Primary))
        .ok_or_else(|| {
            py_value_error(
                "sphere_basis_with_centers: primary penalty was not built; check spec".to_string(),
            )
        })?;
    let penalty = built.penalties[primary_idx].clone();
    let design = built.design.to_dense();
    Ok((
        design.into_pyarray(py).unbind(),
        penalty.into_pyarray(py).unbind(),
    ))
}

/// Resolve `(method, wahba_kernel)` from the user `kernel` string, shared by
/// the sphere basis + sphere jet entry points. Harmonic carries no Wahba
/// kernel; the degree probe is supplied separately by each caller.
fn sphere_kernel_kind_from_str(
    kernel: &str,
    site: &str,
) -> PyResult<(SphereMethod, SphereWahbaKernel)> {
    match kernel.to_ascii_lowercase().as_str() {
        "sobolev" => Ok((SphereMethod::Wahba, SphereWahbaKernel::Sobolev)),
        "pseudo" => Ok((SphereMethod::Wahba, SphereWahbaKernel::Pseudo)),
        "harmonic" => Ok((SphereMethod::Harmonic, SphereWahbaKernel::Sobolev)),
        other => Err(py_value_error(format!(
            "{site} kernel must be one of 'sobolev', 'pseudo', 'harmonic'; got '{other}'"
        ))),
    }
}

/// Analytic DESIGN jet `∂Φ/∂(lat, lon)` of the spherical-spline basis built by
/// `sphere_basis` (auto Wahba farthest-point centers, or harmonic degree `L =
/// n_centers`).
///
/// Returns a `(N, K, 2)` array where `K` equals the column count of the
/// `sphere_basis` design and the last axis is `(∂col/∂lat, ∂col/∂lon)` in the
/// same angular units as the input (degrees by default, radians when
/// `radians=True`). All derivatives are exact analytic forms — no finite
/// differences.
#[pyfunction(signature = (points, n_centers, penalty_order = 2, kernel = "sobolev", radians = false))]
fn sphere_basis_jet<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    n_centers: usize,
    penalty_order: usize,
    kernel: &str,
    radians: bool,
) -> PyResult<Py<PyArray3<f64>>> {
    let pts = points.as_array();
    if pts.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_basis_jet expects points of shape (N, 2) [lat, lon]; got d={}",
            pts.ncols()
        )));
    }
    if !(1..=4).contains(&penalty_order) {
        return Err(py_value_error(format!(
            "sphere_basis_jet penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        )));
    }
    let (method, wahba_kernel) = sphere_kernel_kind_from_str(kernel, "sphere_basis_jet")?;
    let max_degree = matches!(method, SphereMethod::Harmonic).then_some(n_centers);
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
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
    };
    let jet = spherical_spline_design_jet(pts, &spec).map_err(basis_error_to_pyerr)?;
    Ok(jet.into_pyarray(py).unbind())
}

/// Analytic DESIGN jet `∂Φ/∂(lat, lon)` of the spherical-spline basis built by
/// `sphere_basis_with_centers` (explicit Wahba centers; harmonic uses
/// `L = centers.nrows()` as a degree probe, mirroring the forward).
///
/// Returns `(N, K, 2)` aligned column-for-column with the
/// `sphere_basis_with_centers` design, last axis `(∂col/∂lat, ∂col/∂lon)`.
#[pyfunction(signature = (points, centers, penalty_order = 2, kernel = "sobolev", radians = false))]
fn sphere_basis_jet_with_centers<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    penalty_order: usize,
    kernel: &str,
    radians: bool,
) -> PyResult<Py<PyArray3<f64>>> {
    let pts = points.as_array();
    let ctrs = centers.as_array();
    if pts.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_basis_jet_with_centers expects points of shape (N, 2) [lat, lon]; got d={}",
            pts.ncols()
        )));
    }
    if ctrs.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_basis_jet_with_centers expects centers of shape (K, 2) [lat, lon]; got d={}",
            ctrs.ncols()
        )));
    }
    if !(1..=4).contains(&penalty_order) {
        return Err(py_value_error(format!(
            "sphere_basis_jet_with_centers penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        )));
    }
    let (method, wahba_kernel) =
        sphere_kernel_kind_from_str(kernel, "sphere_basis_jet_with_centers")?;
    let max_degree = matches!(method, SphereMethod::Harmonic).then_some(ctrs.nrows());
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::UserProvided(ctrs.to_owned()),
        penalty_order,
        double_penalty: false,
        radians,
        method,
        max_degree,
        wahba_kernel,
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
    };
    let jet = spherical_spline_design_jet(pts, &spec).map_err(basis_error_to_pyerr)?;
    Ok(jet.into_pyarray(py).unbind())
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
    // The chart-local sphere basis, its lat/lon jet, and the saturated-latitude
    // `chain_lat` gating all live in the core SAE path; this helper only routes
    // the caller's coordinates through that single source of truth and converts
    // the returned arrays into Python-facing buffers. Keeping the math in one
    // place is what prevents the core and PyFFI derivatives from drifting.
    let coords = t.as_array();
    let (phi, jet) = sphere_chart_basis_jet(coords).map_err(py_value_error)?;
    let penalty = Array2::from_diag(&Array1::from_vec(SPHERE_CHART_PENALTY_DIAGONAL.to_vec()));
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
        .map_err(basis_error_to_pyerr)?;
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
    // Use the typed detacher so any `EstimationError` raised by
    // `gaussian_reml_free_b_score` reaches Python as its specific
    // subclass (RemlConvergenceError, IllConditionedError, …) instead
    // of being flattened to a generic ValueError. The `apply_by_gate`
    // input-validation errors are wrapped into `EstimationError::InvalidInput`
    // so the closure's error type stays uniform.
    let score = detach_estimation_result(py, "gaussian_reml_score", move || {
        let gated_x = gate_design_for_forward(
            x_values.view(),
            by_values.as_ref().map(|b| b.view()),
            by_start_col,
        )
        .map_err(EstimationError::InvalidInput)?;
        let fit_x = gated_x.as_ref().map_or(x_values.view(), |g| g.view());
        let gated_weights = gate_weights_for_forward(
            weight_values.as_ref().map(|w| w.view()),
            by_values.as_ref().map(|b| b.view()),
            x_values.nrows(),
        )
        .map_err(EstimationError::InvalidInput)?;
        gaussian_reml_free_b_score(
            fit_x,
            y_values.view(),
            coefficient_values.view(),
            log_lambda,
            penalty_values.view(),
            gated_weights.as_ref().map(|w| w.view()),
        )
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
        return Err(py_value_error(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

#[pyfunction(signature = (y_out, y_hat, z, w_dec, lambda_sparse, skip_u = None, skip_proj = None))]
fn skip_transcoder_reml_metrics<'py>(
    py: Python<'py>,
    y_out: PyReadonlyArray2<'py, f64>,
    y_hat: PyReadonlyArray2<'py, f64>,
    z: PyReadonlyArray2<'py, f64>,
    w_dec: PyReadonlyArray2<'py, f64>,
    lambda_sparse: f64,
    skip_u: Option<PyReadonlyArray2<'py, f64>>,
    skip_proj: Option<PyReadonlyArray2<'py, f64>>,
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
    let skip_proj_view = skip_proj.as_ref().map(|value| value.as_array());

    require_finite_matrix("y_out", &y_out)?;
    require_finite_matrix("y_hat", &y_hat)?;
    require_finite_matrix("z", &z)?;
    require_finite_matrix("w_dec", &w_dec)?;
    if let Some(skip_u) = skip_u_view.as_ref() {
        require_finite_matrix("skip_u", skip_u)?;
    }
    if let Some(skip_proj) = skip_proj_view.as_ref() {
        require_finite_matrix("skip_proj", skip_proj)?;
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
    // The skip bypass enters the prediction only through the products
    // `XV = x_in · skip_V` (passed as `skip_proj`) and `U = skip_u`. The
    // gauge-invariant score requires *both*: `skip_proj` supplies the skip's
    // data-dependent activation columns and `skip_u` its output loadings.
    // Reject any partial specification at this boundary rather than silently
    // dropping a factor and scoring a different object.
    match (skip_u_view.as_ref(), skip_proj_view.as_ref()) {
        (Some(skip_u), Some(skip_proj)) => {
            if skip_u.nrows() != out_dim {
                return Err(py_value_error(format!(
                    "skip_u row mismatch: expected {out_dim}, got {}",
                    skip_u.nrows()
                )));
            }
            if skip_proj.nrows() != n_rows {
                return Err(py_value_error(format!(
                    "skip_proj row mismatch: expected {n_rows}, got {}",
                    skip_proj.nrows()
                )));
            }
            if skip_proj.ncols() != skip_u.ncols() {
                return Err(py_value_error(format!(
                    "skip rank mismatch: skip_u has {} columns but skip_proj has {}",
                    skip_u.ncols(),
                    skip_proj.ncols()
                )));
            }
        }
        (Some(_), None) => {
            return Err(py_value_error(
                "skip_u was provided without skip_proj (= x_in · skip_V); \
                 both factors are required to score the skip bypass"
                    .to_string(),
            ));
        }
        (None, Some(_)) => {
            return Err(py_value_error(
                "skip_proj was provided without skip_u; both factors are \
                 required to score the skip bypass"
                    .to_string(),
            ));
        }
        (None, None) => {}
    }

    let metrics = skip_transcoder_reml_metrics_core(SkipTranscoderRemlInputs {
        y_out,
        y_hat,
        z,
        w_dec,
        lambda_sparse,
        skip_proj: skip_proj_view,
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

const PREFERRED_PREDICTION_COLUMNS: &[&str] = &[
    "linear_predictor",
    "mean",
    "std_error",
    "mean_lower",
    "mean_upper",
    // Response-scale observation (prediction) interval, emitted only when
    // `observation_interval=True` and the family supports it; ordered after
    // the credible mean interval so the standard schema stays stable when off.
    "observation_lower",
    "observation_upper",
    // Issue #365: location-scale / GAMLSS families emit the fitted per-row
    // distribution scale (e.g. Gaussian σ) so the learned `noise_formula`
    // function is retrievable from Python; ordered after the mean columns.
    "noise_scale",
];

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

/// Finalize topology candidate lifecycles through the typed Rust selector.
///
/// Python supplies exactly one terminal outcome per declared candidate:
/// assembly/fit failures, or metadata from one completed fit. Rust owns score
/// construction, evidence validation, failure conversion, deterministic
/// ordering, winner selection, and cross-score disagreement diagnostics.
#[pyfunction]
fn select_topology_candidate_lifecycle(request_json: &str) -> PyResult<String> {
    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum ScoreKind {
        Reml,
        Laml,
        Bic,
        Tk,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum ScoreScale {
        Raw,
        PerObservation,
        PerEffectiveDim,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum FailureStage {
        Assembly,
        Fit,
        Evidence,
    }
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum LifecycleFloat {
        Finite(f64),
        NonFinite(String),
    }
    impl LifecycleFloat {
        fn decode(self) -> Result<f64, String> {
            match self {
                Self::Finite(value) => Ok(value),
                Self::NonFinite(token) => match token.as_str() {
                    "nan" => Ok(f64::NAN),
                    "infinity" => Ok(f64::INFINITY),
                    "-infinity" => Ok(f64::NEG_INFINITY),
                    _ => Err(format!("invalid lifecycle float token {token:?}")),
                },
            }
        }
    }
    #[derive(Deserialize)]
    #[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
    enum CandidateOutcome {
        Fitted {
            name: String,
            raw_reml: LifecycleFloat,
            laml: Option<LifecycleFloat>,
            deviance: Option<LifecycleFloat>,
            null_dim: Option<LifecycleFloat>,
            null_space_logdet: Option<LifecycleFloat>,
            effective_dim: LifecycleFloat,
            basis_size: usize,
            n_obs: usize,
        },
        Failed {
            name: String,
            stage: FailureStage,
            error_type: String,
            message: String,
            evidence_at_failure: Option<LifecycleFloat>,
        },
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LifecycleRequest {
        score_kind: ScoreKind,
        score_scale: ScoreScale,
        candidates: Vec<CandidateOutcome>,
    }

    let request: LifecycleRequest = serde_json::from_str(request_json).map_err(|err| {
        py_value_error(format!(
            "select_topology_candidate_lifecycle: failed to parse request JSON: {err}"
        ))
    })?;
    let score_kind = match request.score_kind {
        ScoreKind::Reml => gam::solver::TopologySelectionScoreKind::Reml,
        ScoreKind::Laml => gam::solver::TopologySelectionScoreKind::Laml,
        ScoreKind::Bic => gam::solver::TopologySelectionScoreKind::Bic,
        ScoreKind::Tk => gam::solver::TopologySelectionScoreKind::Tk,
    };
    let score_scale = match request.score_scale {
        ScoreScale::Raw => gam::solver::TopologySelectionScoreScale::Raw,
        ScoreScale::PerObservation => gam::solver::TopologySelectionScoreScale::PerObservation,
        ScoreScale::PerEffectiveDim => gam::solver::TopologySelectionScoreScale::PerEffectiveDim,
    };
    let candidates: Result<Vec<_>, String> = request
        .candidates
        .into_iter()
        .map(|candidate| match candidate {
            CandidateOutcome::Fitted {
                name,
                raw_reml,
                laml,
                deviance,
                null_dim,
                null_space_logdet,
                effective_dim,
                basis_size,
                n_obs,
            } => Ok(gam::solver::TopologyCandidateOutcome::Fitted(
                gam::solver::TopologyCandidateEvidence {
                    name,
                    raw_reml: raw_reml.decode()?,
                    laml: laml.map(LifecycleFloat::decode).transpose()?,
                    deviance: deviance.map(LifecycleFloat::decode).transpose()?,
                    null_dim: null_dim.map(LifecycleFloat::decode).transpose()?,
                    null_space_logdet: null_space_logdet.map(LifecycleFloat::decode).transpose()?,
                    effective_dim: effective_dim.decode()?,
                    basis_size,
                    n_obs,
                },
            )),
            CandidateOutcome::Failed {
                name,
                stage,
                error_type,
                message,
                evidence_at_failure,
            } => Ok(gam::solver::TopologyCandidateOutcome::Failed(
                gam::solver::TopologyCandidateFailure {
                    name,
                    stage: match stage {
                        FailureStage::Assembly => {
                            gam::solver::TopologyCandidateFailureStage::Assembly
                        }
                        FailureStage::Fit => gam::solver::TopologyCandidateFailureStage::Fit,
                        FailureStage::Evidence => {
                            gam::solver::TopologyCandidateFailureStage::Evidence
                        }
                    },
                    error_type,
                    message,
                    evidence_at_failure: evidence_at_failure
                        .map(LifecycleFloat::decode)
                        .transpose()?,
                },
            )),
        })
        .collect();
    let candidates = candidates.map_err(|err| {
        py_value_error(format!(
            "select_topology_candidate_lifecycle: invalid numeric payload: {err}"
        ))
    })?;
    let selected =
        gam::solver::select_topology_candidate_lifecycle(candidates, score_kind, score_scale)
            .map_err(|err| py_value_error(format!("select_topology_candidate_lifecycle: {err}")))?;
    let ranked: Vec<serde_json::Value> = selected
        .ranked
        .into_iter()
        .map(|row| {
            serde_json::json!({
                "name": row.name,
                "score": row.score,
                "raw_reml": row.raw_reml,
                "effective_dim": row.effective_dim,
                "basis_size": row.basis_size,
                "n_obs": row.n_obs,
            })
        })
        .collect();
    let failed: Vec<serde_json::Value> = selected
        .failed
        .into_iter()
        .map(|failure| {
            serde_json::json!({
                "name": failure.name,
                "stage": failure.stage.as_str(),
                "error_type": failure.error_type,
                "message": failure.message,
                "evidence_at_failure": failure.evidence_at_failure,
            })
        })
        .collect();
    serde_json::to_string(&serde_json::json!({
        "ranked": ranked,
        "winner_index": selected.winner_index,
        "failed": failed,
        "warnings": selected.warnings,
    }))
    .map_err(|err| {
        py_value_error(format!(
            "select_topology_candidate_lifecycle: serialise: {err}"
        ))
    })
}

/// Solve the stacking-of-predictive-distributions weight problem over retained
/// topology candidates (#768). `names` aligns with the columns of the
/// row-major held-out log-predictive-density table `log_density_rows` (each
/// inner vector is one held-out observation row over candidates). Returns a
/// JSON object `{ "weights": {name: w}, "mean_log_score": f, "iterations": k }`
/// where the weights are the simplex maximiser of the held-out mean log-score.
/// Candidates with no finite held-out density are rejected and zero-weighted.
#[pyfunction]
fn stacking_weights_from_log_density(
    names: Vec<String>,
    log_density_rows: Vec<Vec<f64>>,
) -> PyResult<String> {
    if names.is_empty() {
        return Err(py_value_error(
            "stacking_weights_from_log_density: at least one candidate name is required"
                .to_string(),
        ));
    }
    let n_cand = names.len();
    if log_density_rows.is_empty() {
        return Err(py_value_error(
            "stacking_weights_from_log_density: at least one held-out row is required".to_string(),
        ));
    }
    let n_rows = log_density_rows.len();
    let mut table = Array2::<f64>::zeros((n_rows, n_cand));
    for (i, row) in log_density_rows.iter().enumerate() {
        if row.len() != n_cand {
            return Err(py_value_error(format!(
                "stacking_weights_from_log_density: row {i} has {} entries but {n_cand} candidates",
                row.len()
            )));
        }
        for (k, &value) in row.iter().enumerate() {
            table[[i, k]] = value;
        }
    }
    let solved = gam::solver::evidence::solve_stacking_weights(
        table.view(),
        gam::solver::evidence::StackingConfig::default(),
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    let weights_by_name: serde_json::Map<String, serde_json::Value> = names
        .iter()
        .zip(solved.weights.iter())
        .map(|(name, &w)| (name.clone(), serde_json::json!(w)))
        .collect();
    let out = serde_json::json!({
        "weights": weights_by_name,
        "mean_log_score": solved.mean_log_score(),
        "iterations": solved.iterations,
    });
    serde_json::to_string(&out).map_err(|err| {
        py_value_error(format!(
            "stacking_weights_from_log_density: serialise: {err}"
        ))
    })
}

/// Topology stacking from the raw per-candidate held-out predictive moments
/// (#768). Migrated CORE-MATH from `gamfit._select_topology.stack_topologies`:
/// recovers each candidate's per-point predictive σ by inverting its
/// `[lower, upper]` observation interval at coverage `interval_level`, forms the
/// held-out Gaussian log-density table, and solves for the simplex stacking
/// weights. `means`, `lowers`, and `uppers` are indexed `[candidate][row]`, one
/// inner list per name. Non-scorable rows (σ ≤ 0 or non-finite mean/σ) carry no
/// density and are dropped by the solve, exactly as the old Python code did.
/// Returns the SAME JSON shape as `stacking_weights_from_log_density`:
/// `{ "weights": {name: w}, "mean_log_score": f, "iterations": k }`.
#[pyfunction]
fn stack_topologies_gaussian(
    py: Python<'_>,
    names: Vec<String>,
    y: Vec<f64>,
    means: Vec<Vec<f64>>,
    lowers: Vec<Vec<f64>>,
    uppers: Vec<Vec<f64>>,
    interval_level: f64,
) -> PyResult<String> {
    if names.is_empty() {
        return Err(py_value_error(
            "stack_topologies_gaussian: at least one candidate name is required".to_string(),
        ));
    }
    if means.len() != names.len() {
        return Err(py_value_error(format!(
            "stack_topologies_gaussian: {} names but {} candidate mean columns",
            names.len(),
            means.len()
        )));
    }
    let solved = py
        .detach(|| {
            gam::solver::topology_stack_gaussian::stack_topologies_gaussian(
                &y,
                &means,
                &lowers,
                &uppers,
                interval_level,
            )
        })
        .map_err(|err| py_value_error(format!("stack_topologies_gaussian: {err}")))?;
    let weights_by_name: serde_json::Map<String, serde_json::Value> = names
        .iter()
        .zip(solved.weights.iter())
        .map(|(name, &w)| (name.clone(), serde_json::json!(w)))
        .collect();
    let out = serde_json::json!({
        "weights": weights_by_name,
        "mean_log_score": solved.mean_log_score(),
        "iterations": solved.iterations,
    });
    serde_json::to_string(&out)
        .map_err(|err| py_value_error(format!("stack_topologies_gaussian: serialise: {err}")))
}

const REML_SCORE_KEYS: &[&str] = &["reml_score", "evidence", "laml", "score"];

const RAW_REML_SCORE_KEYS: &[&str] = &["raw_reml_score"];

const EDF_KEYS: &[&str] = &["edf_total", "edf", "effective_dof"];

const LOG_LIK_KEYS: &[&str] = &["log_likelihood", "loglik", "log_lik"];
// Response-family tag, used only by the compare_models comparability guard
// (#1384). `family_name` is the SummaryPayload field; the aliases cover a
// dict / .evidence object that exposes it under a shorter key.
const FAMILY_KEYS: &[&str] = &["family_name", "family"];
// Observation count, used only by the compare_models comparability guard. `n_obs`
// is the SummaryPayload field; the aliases cover a dict / .evidence object that
// exposes it under a longer name.
const NUM_OBS_KEYS: &[&str] = &["n_obs", "n_observations", "num_observations", "nobs"];

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
        return Err(PyValueError::new_err(
            "compare_models requires at least one fit",
        ));
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

    // Python-specific work: extract scalar score + edf from each PyAny
    // fit (which may be a saved-summary dict, a Model object, or any
    // object exposing .evidence). Then the ranking, delta, Bayes-factor,
    // and evidence-summary logic is delegated to the pure-Rust core in
    // `gam::solver::evidence`, which is identically callable from
    // the CLI binary.
    let mut candidates = Vec::with_capacity(fits.len());
    for (index, (name, fit)) in labels.into_iter().zip(fits.iter()).enumerate() {
        let fit = fit.bind(py);
        let view = reml_fit_view(py, fit)?;
        candidates.push(RemlCandidate {
            index,
            name,
            score: extract_reml_score_from_view(py, &view)?,
            edf: extract_edf_from_view(py, &view)?,
            log_lik: extract_log_lik_from_view(&view)?,
            family: extract_family_from_view(&view)?,
            n_obs: extract_n_obs_from_view(&view)?,
        });
    }

    let comparison = compare_reml_fits_core(candidates.clone()).map_err(PyValueError::new_err)?;

    let ranking = PyList::empty(py);
    for row in comparison.ranking.iter() {
        ranking.append((
            row.name.as_str(),
            row.score,
            row.delta,
            row.bayes_factor,
            row.edf,
        ))?;
    }
    let score_table = PyList::empty(py);
    for row in comparison.score_table.iter() {
        let table_row = PyDict::new(py);
        table_row.set_item("name", row.name.as_str())?;
        table_row.set_item("reml_score", row.reml_score)?;
        table_row.set_item("delta_reml", row.delta_reml)?;
        table_row.set_item(
            "bayes_factor_best_over_model",
            row.bayes_factor_best_over_model,
        )?;
        table_row.set_item("effective_dof", row.effective_dof)?;
        score_table.append(table_row)?;
    }

    let out = PyDict::new(py);
    out.set_item("ranking", ranking)?;
    out.set_item("winner", &comparison.winner)?;
    out.set_item("evidence_summary", &comparison.evidence_summary)?;
    out.set_item("score_table", score_table)?;
    if let Some(scores) = cv_scores {
        // cv_optional walks the ranked order but uses the caller's
        // original score indices — preserved via `RemlCandidate.index`.
        let by_name: std::collections::HashMap<&str, usize> = candidates
            .iter()
            .map(|c| (c.name.as_str(), c.index))
            .collect();
        let cv_optional = PyList::empty(py);
        for row in comparison.ranking.iter() {
            let original_index = by_name[row.name.as_str()];
            cv_optional.append((row.name.as_str(), scores[original_index]))?;
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
    if let Some(score) = extract_float_metadata_from_view(view, RAW_REML_SCORE_KEYS)? {
        return Ok(score);
    }
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
    comparable_reml_score(
        score,
        Some(null_dim),
        extract_float_metadata_from_view(view, NULL_HESSIAN_LOGDET_KEYS)?,
    )
    .map_err(PyValueError::new_err)
}

fn comparable_reml_score(
    raw_reml_score: f64,
    null_dim: Option<f64>,
    null_space_logdet: Option<f64>,
) -> Result<f64, String> {
    let Some(null_dim) = null_dim else {
        return Ok(raw_reml_score);
    };
    gam::solver::topology_selector::tk_normalized_score(
        raw_reml_score,
        null_dim,
        null_space_logdet,
        1.0,
        1,
        gam::solver::evidence::TopologyScoreScale::PerObservation,
    )
}

/// Occam-penalised conditional-AIC ranking score for a saved-model summary
/// payload, matching `gam::solver::evidence::RemlCandidate::ranking_score`
/// exactly (`-2·loglik + 2·edf`) so `Model.evidence` and `Model.bayes_factor_vs`
/// pick the SAME winner as `gamfit.compare_models` (issue #2079).
///
/// `compare_models` ranks on this conditional AIC (which prices the effective
/// degrees of freedom a pure-noise smooth spends, penalising it correctly),
/// while `evidence` / `bayes_factor_vs` previously used the raw REML/LAML
/// evidence headline (`comparable_reml_score_from_summary_payload`) — the two
/// picked OPPOSITE winners when a model was augmented with a near-null smooth.
/// Routing both through this helper removes that cross-method contradiction.
///
/// Falls back to the raw comparable REML/LAML score when the payload predates
/// the log-likelihood / edf fields (both must be present and finite), matching
/// the `ranking_score` fallback in `evidence.rs`.
fn ranking_score_from_summary_payload(payload: &serde_json::Value) -> PyResult<f64> {
    let log_lik = json_lookup_f64(payload, LOG_LIK_KEYS);
    let edf = json_lookup_edf(payload, EDF_KEYS);
    if let (Some(log_lik), Some(edf)) = (log_lik, edf) {
        if log_lik.is_finite() && edf.is_finite() {
            return Ok(-2.0 * log_lik + 2.0 * edf);
        }
    }
    comparable_reml_score_from_summary_payload(payload)
}

fn comparable_reml_score_from_summary_payload(payload: &serde_json::Value) -> PyResult<f64> {
    let raw = json_lookup_f64(payload, RAW_REML_SCORE_KEYS)
        .or_else(|| json_lookup_f64(payload, REML_SCORE_KEYS))
        .ok_or_else(|| py_value_error("saved model payload is missing reml_score".to_string()))?;
    if !raw.is_finite() {
        return Err(py_value_error(
            "saved model payload reml_score must be finite".to_string(),
        ));
    }
    comparable_reml_score(
        raw,
        json_lookup_f64(payload, NULL_DIM_KEYS),
        json_lookup_f64(payload, NULL_HESSIAN_LOGDET_KEYS),
    )
    .map_err(PyValueError::new_err)
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

/// Log-likelihood at the converged mode, used by `compare_models` to form the
/// Occam-penalised conditional AIC that decides the winner (issue #1362).
/// `None` when the fit payload predates the field — the comparison then falls
/// back to the raw REML/LAML evidence headline.
fn extract_log_lik_from_view(view: &RemlFitView<'_>) -> PyResult<Option<f64>> {
    extract_float_metadata_from_view(view, LOG_LIK_KEYS)
}

/// Response-family tag of a candidate fit, for the compare_models comparability
/// guard (#1384). `None` when the fit does not expose one (legacy payloads),
/// which the guard treats as unconstrained.
fn extract_family_from_view(view: &RemlFitView<'_>) -> PyResult<Option<String>> {
    extract_string_metadata_from_view(view, FAMILY_KEYS)
}

/// Observation count of a candidate fit, for the compare_models cross-`n`
/// comparability guard. `None` when the fit does not expose one (legacy payloads
/// / O(n) scan smoothers), which the guard treats as unconstrained. A
/// non-finite or negative value is dropped to `None` rather than truncated.
fn extract_n_obs_from_view(view: &RemlFitView<'_>) -> PyResult<Option<usize>> {
    Ok(extract_float_metadata_from_view(view, NUM_OBS_KEYS)?
        .filter(|value| value.is_finite() && *value >= 1.0)
        .map(|value| value as usize))
}

/// String-valued metadata lookup over the same SavedSummary-JSON / PythonObject
/// fallback chain as [`extract_float_metadata_from_view`].
fn extract_string_metadata_from_view(
    view: &RemlFitView<'_>,
    keys: &[&str],
) -> PyResult<Option<String>> {
    match view {
        RemlFitView::SavedSummary(payload) => Ok(json_lookup_str(payload, keys)),
        RemlFitView::PythonObject(_) => {
            let Some(value) = extract_py_metadata_value(view, keys)? else {
                return Ok(None);
            };
            value.extract::<String>().map(Some)
        }
    }
}

/// First string value found under any of `keys` in a SavedSummary JSON payload.
fn json_lookup_str(payload: &serde_json::Value, keys: &[&str]) -> Option<String> {
    let object = payload.as_object()?;
    for key in keys {
        if let Some(value) = object.get(*key) {
            if let Some(s) = value.as_str() {
                return Some(s.to_string());
            }
        }
    }
    None
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

fn reml_fit_view<'py>(py: Python<'py>, fit: &Bound<'py, PyAny>) -> PyResult<RemlFitView<'py>> {
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

#[pyfunction(signature = (x, y, penalty, weights, ridge_lambda))]
fn gaussian_weighted_ridge_array<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    ridge_lambda: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let (coefficients, fitted) = gam::linalg::utils::gaussian_weighted_ridge(
        x.as_array(),
        y.as_array(),
        penalty.as_array(),
        weights.as_array(),
        ridge_lambda,
    )
    .map_err(py_value_error)?;
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
    let (coefficients, fitted) = gam::linalg::utils::gaussian_weighted_ridge_batch(
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

/// Batched closed-form analytic VJP (reverse-mode adjoint) of the Gaussian
/// row-weighted ridge solve. Single Rust source of truth for the torch
/// `_GaussianWeightedRidge*Fn` backward: given the upstream cotangents
/// `grad_coef` (wrt `coef`, `(K,M,D)`) and `grad_fitted` (wrt `fitted`,
/// `(K,Nmax,D)`), returns the gradients wrt `X (K,Nmax,M)`, `Y (K,Nmax,D)`,
/// `penalty (M,M)` (summed across problems) and `weights (K,Nmax)`. Padded rows
/// (index `>= row_counts[k]`) contribute exactly zero, matching the forward's
/// active-prefix solve. The single-problem torch path routes through here with a
/// leading batch axis of one. See
/// `gam::linalg::gaussian_weighted_ridge_backward::gaussian_weighted_ridge_batch_backward`.
#[pyfunction]
#[pyo3(signature = (grad_coef, grad_fitted, x, y, penalty, weights, coef, ridge_lambda, row_counts = None))]
fn gaussian_weighted_ridge_batch_backward<'py>(
    py: Python<'py>,
    grad_coef: PyReadonlyArray3<'py, f64>,
    grad_fitted: PyReadonlyArray3<'py, f64>,
    x: PyReadonlyArray3<'py, f64>,
    y: PyReadonlyArray3<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, f64>,
    coef: PyReadonlyArray3<'py, f64>,
    ridge_lambda: f64,
    row_counts: Option<PyReadonlyArray1<'py, usize>>,
) -> PyResult<(
    Py<PyArray3<f64>>,
    Py<PyArray3<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
)> {
    let grad_coef_owned = grad_coef.as_array().to_owned();
    let grad_fitted_owned = grad_fitted.as_array().to_owned();
    let x_owned = x.as_array().to_owned();
    let y_owned = y.as_array().to_owned();
    let penalty_owned = penalty.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let coef_owned = coef.as_array().to_owned();
    let row_counts_owned = row_counts.map(|counts| counts.as_array().to_owned());
    let (grad_x, grad_y, grad_penalty, grad_weights) = py
        .detach(move || {
            gam::linalg::gaussian_weighted_ridge_backward::gaussian_weighted_ridge_batch_backward(
                grad_coef_owned.view(),
                grad_fitted_owned.view(),
                x_owned.view(),
                y_owned.view(),
                penalty_owned.view(),
                weights_owned.view(),
                coef_owned.view(),
                ridge_lambda,
                row_counts_owned.as_ref().map(|counts| counts.view()),
            )
        })
        .map_err(py_value_error)?;
    Ok((
        grad_x.into_pyarray(py).unbind(),
        grad_y.into_pyarray(py).unbind(),
        grad_penalty.into_pyarray(py).unbind(),
        grad_weights.into_pyarray(py).unbind(),
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
        let gated_x = gate_design_for_forward(
            x_values.view(),
            by_values.as_ref().map(|b| b.view()),
            by_start_col,
        )?;
        let fit_x = gated_x.as_ref().map_or(x_values.view(), |g| g.view());
        let gated_weights = gate_weights_for_forward(
            weight_values.as_ref().map(|w| w.view()),
            by_values.as_ref().map(|b| b.view()),
            x_values.nrows(),
        )?;
        match gaussian_reml_multi_closed_form_with_cache(
            fit_x,
            y_values.view(),
            penalty_values.view(),
            gated_weights.as_ref().map(|w| w.view()),
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
        detach_estimation_result(py, "gaussian_reml_fit_backward", move || {
            let gated_x = gate_design_for_forward(
                x_values.view(),
                by_values.as_ref().map(|b| b.view()),
                by_start_col,
            )
            .map_err(EstimationError::InvalidInput)?;
            let fit_x = gated_x.as_ref().map_or(x_values.view(), |g| g.view());
            let gated_weights = gate_weights_for_forward(
                weight_values.as_ref().map(|w| w.view()),
                by_values.as_ref().map(|b| b.view()),
                x_values.nrows(),
            )
            .map_err(EstimationError::InvalidInput)?;
            let backward = if let Some(fit) = forward_fit.as_ref() {
                gaussian_reml_multi_closed_form_backward_from_fit(
                    fit_x,
                    y_values.view(),
                    penalty_values.view(),
                    gated_weights.as_ref().map(|w| w.view()),
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
                    gated_weights.as_ref().map(|w| w.view()),
                    init_lambda,
                    grad_lambda,
                    grad_coefficients_values.as_ref().map(|g| g.view()),
                    grad_fitted_values.as_ref().map(|g| g.view()),
                    grad_reml_score,
                    grad_edf,
                )
            }?;
            let (grad_x, grad_by) = ungate_design_gradient(
                x_values.view(),
                by_values.as_ref().map(|b| b.view()),
                by_start_col,
                backward.grad_x,
            )
            .map_err(EstimationError::InvalidInput)?;
            let grad_weights =
                ungate_weight_gradient(by_values.as_ref().map(|b| b.view()), backward.grad_weights);
            Ok((
                grad_x,
                grad_by,
                backward.grad_y,
                backward.grad_penalty,
                grad_weights,
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
    rows: PyRef<'py, PyEncodedTable>,
    formula: String,
    y: PyReadonlyArray2<'py, f64>,
    config_json: Option<String>,
    fisher_rao_w: Option<PyReadonlyArray3<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let y_values = y.as_array().to_owned();
    let fisher_values = fisher_rao_w.as_ref().map(|w| w.as_array().to_owned());
    let result = detach_typed_py_result(
        py,
        "gaussian_reml_fit_formula_table",
        move || {
            gaussian_reml_fit_formula_dataset_impl(
                dataset,
                formula,
                y_values.view(),
                config_json.as_deref(),
                fisher_values.as_ref().map(|w| w.view()),
            )
        },
        |_, error| match error {
            SharedTangentFfiError::Spec(message) => py_value_error(message),
            SharedTangentFfiError::Engine(engine) => estimation_error_to_pyerr(engine),
        },
    )?;
    tangent_reml_result_to_pydict(py, result)
}

fn tangent_reml_result_to_pydict<'py>(
    py: Python<'py>,
    fit: TangentRemlMultiResult,
) -> PyResult<Py<PyDict>> {
    let finite = fit.reml_score.is_finite()
        && fit.coefficients.iter().all(|value| value.is_finite())
        && fit.lambdas.iter().all(|value| value.is_finite());
    let out = PyDict::new(py);
    out.set_item("status", if finite { "ok" } else { "diverged" })?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("coefficients", fit.coefficients.into_pyarray(py))?;
    out.set_item("fitted", fit.fitted.into_pyarray(py))?;
    out.set_item("sigma2", fit.sigma2.into_pyarray(py))?;
    out.set_item("lambdas", fit.lambdas.into_pyarray(py))?;
    out.set_item("edf", fit.edf.into_pyarray(py))?;
    Ok(out.unbind())
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
    if n_rows == 0 || p_total == 0 {
        return Err(py_value_error(
            "gaussian_reml_fit_blocks_forward requires non-empty rows and at least one coefficient column"
                .to_string(),
        ));
    }
    let mut joint_x = Array2::<f64>::zeros((n_rows, p_total));
    for (i, d) in designs.iter().enumerate() {
        joint_x
            .slice_mut(s![.., col_offsets[i]..col_offsets[i + 1]])
            .assign(&d.as_array());
    }

    let mut s_list: Vec<gam::terms::smooth::BlockwisePenalty> = Vec::with_capacity(designs.len());
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
        s_list.push(gam::terms::smooth::BlockwisePenalty::new(
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
            Some(rv.iter().copied().collect())
        }
        None => None,
    };

    let offset_zero = Array1::<f64>::zeros(n_rows);
    let opts = gam::solver::estimate::FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
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
        persist_warm_start_disk: false,
    };
    let joint_x_for_fit = joint_x.clone();
    let fit = detach_estimation_result(py, "gaussian_reml_fit_blocks_forward", move || {
        let heuristic_slice = heuristic_owned.as_ref().map(|values| values.as_slice());
        gam::solver::estimate::fit_gamwith_heuristic_lambdas(
            joint_x_for_fit,
            y_col.view(),
            weights_owned.view(),
            offset_zero.view(),
            &s_list,
            heuristic_slice,
            LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
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

#[pyfunction(signature = (designs, penalties, y, weights = None, init_rhos = None))]
fn gaussian_reml_fit_blocks_orthogonal_forward<'py>(
    py: Python<'py>,
    designs: Vec<PyReadonlyArray2<'py, f64>>,
    penalties: Vec<PyReadonlyArray2<'py, f64>>,
    y: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_rhos: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    let designs_owned = designs
        .iter()
        .map(|design| design.as_array().to_owned())
        .collect::<Vec<_>>();
    let penalties_owned = penalties
        .iter()
        .map(|penalty| penalty.as_array().to_owned())
        .collect::<Vec<_>>();
    let y_owned = y.as_array().to_owned();
    let weights_owned = weights
        .as_ref()
        .map(|weights| weights.as_array().to_owned());
    let init_owned = init_rhos.as_ref().map(|rhos| rhos.as_array().to_vec());
    let fit = detach_estimation_result(
        py,
        "gaussian_reml_fit_blocks_orthogonal_forward",
        move || {
            gaussian_reml_blocks_orthogonal_shared_scale(
                &designs_owned,
                &penalties_owned,
                y_owned.view(),
                weights_owned.as_ref().map(|weights| weights.view()),
                init_owned.as_deref(),
            )
        },
    )?;
    let out = PyDict::new(py);
    let coef_list = PyList::empty(py);
    for coef in fit.coefficients {
        coef_list.append(coef.into_pyarray(py))?;
    }
    out.set_item("coefficients", coef_list)?;
    out.set_item("fitted", fit.fitted.into_pyarray(py))?;
    out.set_item("lambdas", fit.lambdas.into_pyarray(py))?;
    out.set_item("log_lambdas", fit.log_lambdas.into_pyarray(py))?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("edf", fit.edf.into_pyarray(py))?;
    Ok(out.unbind())
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
        gam::solver::gaussian_reml::gaussian_reml_fit_blocks_backward_analytic(
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

    // Build the constraint payload. An empty A (0 rows) is treated as "no
    // constraint" — same convention used internally when no shape constraint
    // is active.
    let constraints_opt: Option<gam::solver::pirls::LinearInequalityConstraints> =
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
                        gam::solver::pirls::LinearInequalityConstraints::new(
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

    let x_owned = x_view.to_owned();
    let y_owned = y_view.to_owned();
    let penalty_owned = penalty_view.to_owned();
    let init_lambda = init_log_lambda.map(f64::exp);
    let fit = detach_estimation_result(
        py,
        "gaussian_reml_fit_with_constraints_forward",
        move || {
            gam::solver::constrained_gaussian_reml::constrained_gaussian_reml_forward(
                gam::solver::constrained_gaussian_reml::ConstrainedGaussianRemlForwardProblem {
                    x: x_owned.view(),
                    y: y_owned.view(),
                    penalty: penalty_owned.view(),
                    weights: Some(weights_owned.view()),
                    constraints: constraints_opt.as_ref(),
                    init_lambda,
                },
            )
        },
    )?;

    let out = PyDict::new(py);
    out.set_item("coefficients", fit.coefficients.into_pyarray(py))?;
    out.set_item("fitted", fit.fitted.into_pyarray(py))?;
    out.set_item("lambda", fit.lambda)?;
    out.set_item("log_lambda", fit.lambda.ln())?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("edf", fit.edf)?;
    out.set_item("active_indices", fit.active_indices.into_pyarray(py))?;
    Ok(out.unbind())
}

/// Analytic backward (VJP) for `gaussian_reml_fit_with_constraints_forward`.
///
/// At an active certificate the accepted face is affine,
/// `A_act β̂ = b_act`, and its tangent space is `null(A_act)`.  The core
/// adjoint retains the full affine `β̂` while replacing the unconstrained
/// response and penalty kernels by
/// `P = Z (ZᵀHZ)⁻¹ Zᵀ` and `Q = Z (ZᵀSZ)⁺ Zᵀ`.  This carries the penalty's
/// affine cross/constant terms exactly; a response-shifted homogeneous solve
/// would not.
///
/// Implementation status:
/// - **Interior cert (empty active set):** the projection `Z = I_p` is the
///   identity, so the tangent-projected VJP coincides with the unconstrained
///   closed-form Gaussian REML backward. This case delegates to
///   `gaussian_reml_multi_closed_form_backward` and produces gradients
///   identical to `gaussian_reml_fit_backward` (round-off agreement).
/// - **Active cert (non-empty active set):** delegate the cached forward state
///   and constraint certificate to the core KKT/envelope adjoint.  It supports
///   nonzero affine bounds and emits a typed `GradientUnavailableError` at a
///   weakly-active boundary where the derivative is genuinely set-valued.
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
    // The active-face core differentiates the accepted affine KKT state and
    // therefore consumes the exact forward coefficients instead of launching
    // a second, potentially different smoothing optimization.
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
        let lambda = log_lambda_at_optimum
            .map(f64::exp)
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| {
                py_value_error(
                    "active constrained REML backward requires the accepted finite log-lambda"
                        .to_string(),
                )
            })?;
        let coefficients = coefficients_at_optimum.as_ref().ok_or_else(|| {
            py_value_error(
                "active constrained REML backward requires coefficients_at_optimum".to_string(),
            )
        })?;
        let a = a_inequality.as_ref().ok_or_else(|| {
            py_value_error("active constrained REML backward requires a_inequality".to_string())
        })?;
        let b = b_inequality.as_ref().ok_or_else(|| {
            py_value_error("active constrained REML backward requires b_inequality".to_string())
        })?;
        let active = active_indices.as_ref().ok_or_else(|| {
            py_value_error("active constrained REML backward requires active_indices".to_string())
        })?;

        let x_owned = x.as_array().to_owned();
        let y_owned = y.as_array().to_owned();
        let penalty_owned = penalty.as_array().to_owned();
        let weights_owned = weights.as_ref().map(|values| values.as_array().to_owned());
        let a_owned = a.as_array().to_owned();
        let b_owned = b.as_array().to_owned();
        let active_owned = active.as_array().to_owned();
        let coefficients_owned = coefficients.as_array().to_owned();
        let grad_coefficients_owned = grad_coefficients
            .as_ref()
            .map(|values| values.as_array().to_owned());
        let grad_fitted_owned = grad_fitted
            .as_ref()
            .map(|values| values.as_array().to_owned());
        let backward = detach_pyresult(
            py,
            "gaussian_reml_fit_with_constraints_backward",
            move || {
                gam::solver::constrained_gaussian_reml::constrained_gaussian_reml_backward(
                    gam::solver::constrained_gaussian_reml::ConstrainedGaussianRemlBackwardProblem {
                        x: x_owned.view(),
                        y: y_owned.view(),
                        penalty: penalty_owned.view(),
                        weights: weights_owned.as_ref().map(|values| values.view()),
                        a_inequality: a_owned.view(),
                        b_inequality: b_owned.view(),
                        active_indices: active_owned.view(),
                        lambda,
                        coefficients: coefficients_owned.view(),
                        grad_coefficients: grad_coefficients_owned
                            .as_ref()
                            .map(|values| values.view()),
                        grad_fitted: grad_fitted_owned.as_ref().map(|values| values.view()),
                        grad_lambda,
                        grad_log_lambda,
                        grad_reml_score,
                        grad_edf,
                    },
                )
                .map_err(estimation_error_to_pyerr)
            },
        )?;
        let out = PyDict::new(py);
        out.set_item("grad_x", backward.grad_x.into_pyarray(py))?;
        out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
        out.set_item("grad_penalty", backward.grad_penalty.into_pyarray(py))?;
        out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
        return Ok(out.unbind());
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
    let backward = detach_pyresult(
        py,
        "gaussian_reml_fit_with_constraints_backward",
        move || {
            // Typed engine path: `EstimationError` → matching `gamfit.*Error`
            // subclass via `estimation_error_to_pyerr` (issue #343).
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
            .map_err(estimation_error_to_pyerr)
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
        let gated_x = gate_design_for_forward(
            x_values.view(),
            by_values.as_ref().map(|b| b.view()),
            by_start_col,
        )?;
        let fit_x = gated_x.as_ref().map_or(x_values.view(), |g| g.view());
        let gated_weights = gate_weights_for_forward(
            weight_values.as_ref().map(|w| w.view()),
            by_values.as_ref().map(|b| b.view()),
            x_values.nrows(),
        )?;
        gaussian_reml_fit_batched_impl(
            fit_x,
            y_values.view(),
            row_offset_values.view(),
            penalty_values.view(),
            gated_weights.as_ref().map(|w| w.view()),
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
            let gated_x = gate_design_for_forward(
                x_values.view(),
                by_values.as_ref().map(|b| b.view()),
                by_start_col,
            )?;
            let fit_x = gated_x.as_ref().map_or(x_values.view(), |g| g.view());
            let gated_weights = gate_weights_for_forward(
                weight_values.as_ref().map(|w| w.view()),
                by_values.as_ref().map(|b| b.view()),
                x_values.nrows(),
            )?;
            let backward = gaussian_reml_fit_batched_backward_impl(
                fit_x,
                y_values.view(),
                row_offset_values.view(),
                penalty_values.view(),
                gated_weights.as_ref().map(|w| w.view()),
                init_lambda,
                grad_lambda_values.as_ref().map(|g| g.view()),
                grad_coefficients_values.as_ref().map(|g| g.view()),
                grad_fitted_values.as_ref().map(|g| g.view()),
                grad_reml_score_values.as_ref().map(|g| g.view()),
                grad_edf_values.as_ref().map(|g| g.view()),
                forward_fits.as_deref(),
            )?;
            let (grad_x, grad_by) = ungate_design_gradient(
                x_values.view(),
                by_values.as_ref().map(|b| b.view()),
                by_start_col,
                backward.grad_x,
            )?;
            let grad_weights =
                ungate_weight_gradient(by_values.as_ref().map(|b| b.view()), backward.grad_weights);
            Ok((
                backward.statuses,
                grad_x,
                grad_by,
                backward.grad_y,
                backward.grad_penalty,
                grad_weights,
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
        let gated_x =
            gate_design_for_forward(x.view(), by_values.as_ref().map(|b| b.view()), by_start_col)?;
        let fit_x = gated_x.as_ref().map_or(x.view(), |g| g.view());
        let gated_weights = gate_weights_for_forward(
            weight_values.as_ref().map(|w| w.view()),
            by_values.as_ref().map(|b| b.view()),
            x.nrows(),
        )?;
        match gaussian_reml_multi_closed_form_with_cache(
            fit_x,
            y_values.view(),
            penalty_values.view(),
            gated_weights.as_ref().map(|w| w.view()),
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
// See `src/terms/latent_coord.rs`.
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
    periodic: Option<&[Option<f64>]>,
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
    // Resolve a fully admissible (nullspace_order, power) for THIS ambient
    // latent dimension. The pure scale-free polyharmonic kernel exists only
    // when 2(p + s) > d; with the requested null space alone (s = 0) this
    // fails whenever 2p <= d — e.g. m = 2 (p = 2) at latent_dim >= 4, which is
    // exactly issue #875. `resolve_duchon_orders` lifts the spectral power s
    // (and, if pure-mode CPD requires it, the null-space order) until the
    // kernel is well-posed for any d, including the even-d `r^{2m-d} log r`
    // log case. The latent forward design assembles no operator penalties
    // (`operator_penalties: Default::default()`), so `max_op = 0`: only the
    // kernel-existence / CPD guards apply, matching every other Duchon entry
    // point which routes through this same resolver.
    let (resolved_nullspace, resolved_power) =
        resolve_duchon_orders(latent_dim, duchon_nullspace_from_m(m), 0, None);
    // When the optimizer retracts the latent coordinates on a PERIODIC manifold
    // (circle / torus), the decoder MUST be a function on that manifold:
    // Φ(θ) = Φ(θ + period) per circular axis, with the kernel distance measured
    // across the seam. We mirror the POSITION periodic-Duchon path exactly —
    // route through `build_duchon_basis_mixed_periodicity_auto`, which sends the
    // 1-D circle to the Bernoulli Green's-function builder (the true PSD circle
    // kernel, gam#580) and a multi-axis torus to the chord-distance polyharmonic
    // builder. `periodic` carries a per-axis optional period (radians, the chart
    // wrap = TAU for circle/torus); a `None` axis is a Euclidean (open) axis.
    // When `periodic` is `None`/all-open the basis stays byte-identical to the
    // open Euclidean construction (euclidean / sphere / matern latent fits).
    let periodic_flags: Option<Vec<bool>> = periodic.and_then(|axes| {
        if axes.len() == latent_dim && axes.iter().any(|p| p.is_some()) {
            Some(axes.iter().map(|p| p.is_some()).collect())
        } else {
            None
        }
    });
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(center_matrix.clone()),
        length_scale: None,
        power: resolved_power as f64,
        nullspace_order: resolved_nullspace,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    let built = if let Some(flags) = periodic_flags {
        // `periodic` is Some with the same arity (checked above). Each periodic
        // axis carries an explicit chart period (TAU); non-periodic axes get a
        // placeholder period (unused by the builder for `!periodic` axes).
        let axes = periodic.expect("periodic_flags is only Some when periodic is Some");
        let periods: Vec<f64> = axes.iter().map(|p| p.unwrap_or(1.0)).collect();
        build_duchon_basis_mixed_periodicity_auto(t_mat.view(), &spec, &flags, Some(&periods))
            .map_err(|err| {
                format!("failed to evaluate periodic N-D Duchon basis for LatentCoord: {err}")
            })?
    } else {
        build_duchon_basis(t_mat.view(), &spec)
            .map_err(|err| format!("failed to evaluate N-D Duchon basis for LatentCoord: {err}"))?
    };
    let design = built
        .design
        .try_to_dense_by_chunks("latent_duchon_design")
        .map_err(|err| format!("failed to evaluate N-D Duchon basis for LatentCoord: {err}"))?;
    Ok((design, t_mat))
}

/// Input-location jet `∂Φ/∂t` of the PERIODIC latent Duchon design, matching the
/// per-manifold forward `build_latent_duchon_design` builds: the 1-D circle
/// routes through the Bernoulli Green's-function design (gam#580) and the
/// multi-axis torus through the chord-distance polyharmonic design. Returns
/// `Ok(None)` when no axis is periodic (the caller then uses the open Euclidean
/// jet, which is correct for euclidean / sphere / matern latents).
///
/// The two branches differentiate the SAME kernel, with the SAME resolved orders
/// and the SAME constraint nullspace `Z`, as the forward — so the returned jet is
/// the exact derivative of the forward design column-for-column. Building the
/// open Euclidean jet here instead (the issue #876 bug) gave a wrong gradient and
/// a column-count mismatch that nulled the outer gradient and collapsed the
/// latent.
fn build_latent_duchon_periodic_jet(
    t_mat: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    m: usize,
    periodic: Option<&[Option<f64>]>,
) -> Result<Option<Array3<f64>>, String> {
    let latent_dim = t_mat.ncols();
    // Mirror `build_latent_duchon_design`'s gate: a per-axis period descriptor of
    // the right arity with at least one periodic axis.
    let axes = match periodic {
        Some(axes) if axes.len() == latent_dim && axes.iter().any(|p| p.is_some()) => axes,
        _ => return Ok(None),
    };
    // Same resolved (nullspace_order, power) the forward design uses for this
    // ambient latent dimension, so the kernel smoothness order and the Bernoulli
    // order (`user_m = duchon_p_from_nullspace_order(resolved_nullspace)`) match.
    let (resolved_nullspace, resolved_power) =
        resolve_duchon_orders(latent_dim, duchon_nullspace_from_m(m), 0, None);

    if latent_dim == 1 {
        // 1-D circle: the forward routes to `build_periodic_duchon_basis_1d`
        // (Bernoulli kernel). `create_duchon_basis_1d_derivative_dense` with
        // `periodic = true, order = 1` differentiates that exact forward — same
        // collapsed centers, same domain wrap, same constant-only constraint
        // nullspace — and returns the dense `(n, kernel_cols + 1)` first
        // derivative `∂Φ/∂t` (the trailing constant column's derivative is 0).
        let period = axes[0].expect("latent_dim == 1 periodic axis carries a period");
        let dphi_dt = create_duchon_basis_1d_derivative_dense(
            t_mat.column(0),
            centers.column(0),
            resolved_power as f64,
            resolved_nullspace,
            true,
            Some(period),
            1,
        )
        .map_err(|err| format!("failed to evaluate periodic latent Duchon jet: {err}"))?;
        let n_rows = dphi_dt.nrows();
        let n_cols = dphi_dt.ncols();
        let mut jet = Array3::<f64>::zeros((n_rows, n_cols, 1));
        jet.slice_mut(s![.., .., 0]).assign(&dphi_dt);
        return Ok(Some(jet));
    }

    // Multi-axis torus: the forward routes to `build_duchon_basis_mixed_periodicity`
    // (chord-distance polyharmonic, pure spectrum, constant-only nullspace). The
    // `build_duchon_basis_design_and_jets` builder reproduces that SAME design and
    // returns its exact chord-embedding jet, so we take its `J` block. The mixed
    // periodicity path requires the pure polyharmonic spectrum (`power = 0`); the
    // resolver returns `power = 0` for the periodic latent configurations, but
    // assert it so a future order change fails loudly rather than silently
    // diverging from the forward.
    if resolved_power != 0 {
        return Err(format!(
            "periodic torus latent Duchon requires pure polyharmonic spectrum (power = 0); \
             resolver returned power = {resolved_power}"
        ));
    }
    let periodic_flags: Vec<bool> = axes.iter().map(|p| p.is_some()).collect();
    let periods: Vec<f64> = axes.iter().map(|p| p.unwrap_or(1.0)).collect();
    let (_phi, jet, _hess) = gam::terms::basis::build_duchon_basis_design_and_jets(
        t_mat,
        centers,
        None,
        0.0,
        resolved_nullspace,
        &periodic_flags,
        &periods,
    )
    .map_err(|err| format!("failed to evaluate periodic torus latent Duchon jet: {err}"))?;
    Ok(Some(jet))
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
    periodic: Option<&[Option<f64>]>,
) -> Result<(Array2<f64>, Array2<f64>, Array3<f64>), String> {
    let basis_kind = latent_basis_kind(basis_kind)?;
    let (design, t_mat) = match basis_kind {
        "duchon" => {
            let (design, t_mat) =
                build_latent_duchon_design(t_flat, n_obs, latent_dim, centers, m, periodic)?;
            // On a PERIODIC latent manifold (circle / torus) the forward design is
            // the periodic Duchon basis (1-D Bernoulli Green's function or the
            // multi-axis chord-distance polyharmonic) — a DIFFERENT kernel and
            // column layout than the open Euclidean Duchon. Its input-location
            // jet must differentiate that SAME periodic forward, not the open
            // Euclidean basis the generic `latent_input_location_jet` builds.
            // Routing the periodic forward through the open jet produced both a
            // wrong gradient direction AND a column-count mismatch (the open jet
            // carries `d+1` polynomial columns vs. the periodic design's single
            // constant column), which made `value_and_grad` fail the
            // design/jet shape check, return `(+∞, None)`, and hand the outer
            // trust region a zero gradient — so the circle/torus optimizer read
            // "stationary" at the start and collapsed every row to one latent
            // coordinate (issue #876). Build the matching periodic jet here and
            // return early, mirroring the per-manifold forward choice exactly.
            if let Some(jet) = build_latent_duchon_periodic_jet(t_mat.view(), centers, m, periodic)?
            {
                if jet.shape()[1] != design.ncols() {
                    return Err(format!(
                        "periodic latent Duchon design/jet column mismatch: design has {}, jet has {}",
                        design.ncols(),
                        jet.shape()[1]
                    ));
                }
                return Ok((design, t_mat, jet));
            }
            (design, t_mat)
        }
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
                nullspace_shrinkage_survived: None,
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
                identifiability: SphericalSplineIdentifiability::CenterSumToZero,
            };
            let built = build_spherical_spline_basis(t_mat.view(), &spec)
                .map_err(|err| format!("failed to evaluate sphere latent basis: {err}"))?;
            let constraint_transform = match &built.metadata {
                gam::terms::basis::BasisMetadata::Sphere {
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

#[cfg(test)]
mod survival_payload_nonfinite_tests {
    use super::{SurvivalPredictionJsonPayload, SurvivalPredictionPayload};
    use std::collections::BTreeMap;

    /// #1564 (bug 1): the REAL survival prediction payload structs must round-trip
    /// the non-finite values a saturated Royston-Parmar tail legitimately carries.
    /// A saturated fit has `Λ(t) = exp(η) = +∞` (with `S(t) = 0`); before the fix
    /// `serde_json` wrote that `+∞` as a bare `null`, and the typed parse then
    /// failed with `invalid type: null, expected f64`. This test serializes the
    /// producer payload and parses it back through the consumer payload — the
    /// exact engine→Python boundary — asserting the `+∞` (and a defensive `NaN`)
    /// survive.
    #[test]
    fn saturated_survival_payload_round_trips_through_real_structs() {
        let mut columns = BTreeMap::new();
        columns.insert("survival_prob".to_string(), vec![0.0, 0.0]);
        columns.insert("failure_prob".to_string(), vec![1.0, 1.0]);

        let payload = SurvivalPredictionPayload {
            class: "survival_prediction",
            model_class: "survival transformation".to_string(),
            likelihood_mode: "transformation".to_string(),
            times: vec![1.0, 2.0],
            // Saturated tail: cumulative hazard overflows to +∞, S(t) = 0.
            hazard: vec![vec![0.5, f64::INFINITY], vec![0.25, 0.0]],
            survival: vec![vec![0.6, 0.0], vec![0.7, 0.0]],
            cumulative_hazard: vec![vec![0.5, f64::INFINITY], vec![0.36, f64::INFINITY]],
            linear_predictor: vec![1.99, 1000.0],
            columns,
            // Defensive: a delta-method SE that blew up to NaN must also survive
            // rather than crash the parse.
            survival_se: Some(vec![vec![0.01, f64::NAN], vec![0.02, f64::NAN]]),
            eta_se: Some(vec![0.1, f64::INFINITY]),
        };

        let json = serde_json::to_string(&payload).expect("serialize must succeed");
        // The pre-fix `null` encoding is gone; the boundary now uses explicit
        // tokens that the consumer can parse.
        assert!(!json.contains("null"), "no bare nulls in payload: {json}");
        assert!(json.contains("\"Infinity\""), "json: {json}");

        let parsed: SurvivalPredictionJsonPayload =
            serde_json::from_str(&json).expect("parse must succeed (was the #1564 failure)");

        assert_eq!(parsed.class, "survival_prediction");
        let cum = parsed.cumulative_hazard.expect("cumulative_hazard present");
        assert!(cum[0][1].is_infinite() && cum[0][1] > 0.0);
        assert!(cum[1][1].is_infinite() && cum[1][1] > 0.0);
        let haz = parsed.hazard.expect("hazard present");
        assert!(haz[0][1].is_infinite());
        assert_eq!(haz[1][1], 0.0);
        let se = parsed.survival_se.expect("survival_se present");
        assert!(se[0][1].is_nan());
        let eta_se = parsed.eta_se.expect("eta_se present");
        assert!(eta_se[1].is_infinite());
        // Finite fields are untouched.
        assert_eq!(parsed.times.expect("times present"), vec![1.0, 2.0]);
        assert_eq!(
            parsed
                .columns
                .expect("columns present")
                .get("survival_prob")
                .unwrap(),
            &vec![0.0, 0.0]
        );
    }
}
