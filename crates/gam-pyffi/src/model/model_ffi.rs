use competing_risks_decode::{
    competing_risks_columns, competing_risks_numeric_list, competing_risks_string_list,
    set_optional_competing_risks_matrix, set_optional_competing_risks_vector,
};

use manifold_pyclasses::{
    CircleManifold, EuclideanManifold, GrassmannManifold, ProductManifold, SpdManifold,
    SphereManifold, StiefelManifold, TorusManifold,
};



use sklearn_metadata::sklearn_fit_metadata;

use gam::families::inference::saved_summary::{
    compare_saved_models, prediction_model_class_label, saved_model_report_input,
    saved_model_summary, saved_models_evidence_ratio, scan_introspection, scan_smooth_label,
};
use gam::families::inference::summary_text::render_summary_text;

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
    /// RNG seed for deterministic chain initialisation.
    seed: Option<u64>,
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
    /// `"conditional"` (H⁻¹ only) or `"smoothing"` (first-order smoothing
    /// correction `H⁻¹ + J Var(ρ̂) Jᵀ`, erroring if it is unavailable).
    /// `None` keeps the engine default (`"smoothing"`). This
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
}

/// Validated, typed fitted model retained by the Python `Model` shell.
///
/// Persistence remains byte-based, but every accessor shares this immutable
/// value instead of reparsing and revalidating the JSON archive per call. `Arc`
/// makes detaching work from the GIL a constant-time ownership transfer without
/// cloning the fitted payload. The summary is derived from the typed model on
/// first use and retained, so summary-backed accessors never rebuild it.
///
/// The saved-model bytes it was compiled from are kept beside it (O(p²), no
/// per-row data), so the handle is a value: two handles are equal when they
/// were compiled from the same saved model, and it pickles and copies as those
/// bytes, recompiled on load.
#[pyclass(module = "gamfit._rust", name = "_FittedModel", frozen)]
struct PyFittedModel {
    model: Arc<FittedModel>,
    source: Arc<[u8]>,
    summary: std::sync::OnceLock<serde_json::Value>,
}

impl PyFittedModel {
    fn compile(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<Self> {
        let (model, source) = detach_typed_py_result(
            py,
            "compile_model",
            move || load_model_impl(&model_bytes).map(|model| (model, model_bytes)),
            saved_model_error_to_pyerr,
        )?;
        Ok(Self {
            model: Arc::new(model),
            source: source.into(),
            summary: std::sync::OnceLock::new(),
        })
    }

    fn summary_value(&self) -> PyResult<&serde_json::Value> {
        if let Some(summary) = self.summary.get() {
            return Ok(summary);
        }
        let summary = summary_payload_value(&self.model).map_err(PyValueError::new_err)?;
        Ok(self.summary.get_or_init(|| summary))
    }
}

#[pymethods]
impl PyFittedModel {
    #[new]
    fn py_new(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<Self> {
        Self::compile(py, model_bytes)
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> (Bound<'py, pyo3::types::PyType>, (Bound<'py, PyBytes>,)) {
        let source = PyBytes::new(slf.py(), &slf.get().source);
        (slf.get_type(), (source,))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Self>()
            .is_ok_and(|other| *other.get().source == *self.source)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.source.hash(&mut hasher);
        hasher.finish()
    }

    #[getter]
    fn formula(&self) -> &str {
        &self.model.payload().formula
    }

    #[getter]
    fn family(&self) -> &str {
        &self.model.payload().family
    }

    #[getter]
    fn used_device(&self) -> bool {
        self.model.payload().used_device
    }

    #[getter]
    fn training_table_kind(&self) -> &str {
        &self.model.training_table_kind
    }

    /// Human-readable inference advisories recorded while the model was fit —
    /// the mgcv-style "k reduced to the data support" / basis-degradation notes
    /// from the cr/cs/sz cap (#1541, #1542), and any other materialization
    /// advisory. The CLI prints these; gamfit surfaces the SAME notes as
    /// `GamInferenceWarning`s and via `model.notes` (#1543).
    #[getter]
    fn inference_notes(&self) -> Vec<String> {
        self.model.payload().inference_notes.clone()
    }

    /// Informational notes recorded while the model was fit: defaults the
    /// engine chose (the auto knot count of a default B-spline, per-margin
    /// tensor sizes). gamfit exposes them via `model.notes` and the summary
    /// but does not warn. Empty for payloads that predate the field.
    #[getter]
    fn informational_notes(&self) -> Vec<String> {
        self.model.payload().informational_notes.clone()
    }

    /// The canonical fine-grained prediction class label — e.g. `"bernoulli
    /// marginal-slope"`, `"survival marginal-slope"`, `"competing risks
    /// survival"`, `"latent survival"`, `"gaussian location-scale"`,
    /// `"transformation-normal"`, or `"standard"`.
    ///
    /// The persisted `model_kind` field is the *coarse* [`ModelKind`] enum,
    /// which collapses distinct model classes onto a single tag, so gamfit's
    /// introspection sets (`is_marginal_slope`, `is_survival`, …) read this
    /// label, derived from the fitted family state — the same authority the
    /// predict payloads and CLI use.
    #[getter]
    fn predict_class_name(&self) -> String {
        prediction_model_class_label(&self.model)
    }
}

/// Parse the public `covariance_mode` string into the engine enum. `None`
/// keeps the engine default (required smoothing-corrected). The vocabulary is
/// owned by `InferenceCovarianceMode::from_str` — the same parser the CLI's
/// `--covariance-mode` uses — so the accepted spellings cannot drift per
/// frontend.
fn parse_covariance_mode(
    raw: Option<&str>,
) -> Result<Option<gam_predict::InferenceCovarianceMode>, String> {
    raw.map(str::parse).transpose()
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
    /// Response-scale point column of this class (`PredictModelClass::point_column`).
    point_column: &'static str,
    /// Point-payload shape of this model (`FittedModel::prediction_point_shape`);
    /// the Python shaper branches on it instead of the class label.
    point_shape: &'static str,
    /// Ordered point columns of a multi-curve point (`expectile_curves`: one
    /// column per expectile level, in increasing level order). Omitted for
    /// single-column points, which `point_column` names.
    #[serde(skip_serializing_if = "Option::is_none")]
    point_columns: Option<Vec<String>>,
    /// Inverse-link family kind tag (`identity`, `logit`, `probit`, `log`, ...).
    family: String,
    /// Provenance of the returned prediction interval (#942). Present only on
    /// the conformal routes, which name the full-conformal or split-conformal
    /// construction they ran. `None` (omitted) for point-only and model-based
    /// predictions.
    #[serde(skip_serializing_if = "Option::is_none")]
    interval_method: Option<String>,
    /// Exact covariance definition used for model-based interval uncertainty.
    /// Omitted for point-only and conformal predictions.
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_source: Option<String>,
    /// Exact covariance definition used to integrate a posterior-mean POINT
    /// prediction (#2296) — conditional by definition on curved links, and a
    /// separate fact from the band's `covariance_source`. Omitted when the
    /// point is a plug-in that consulted no coefficient covariance.
    #[serde(skip_serializing_if = "Option::is_none")]
    point_covariance_source: Option<String>,
    /// What a posterior-mean POINT is conditional on when the fit withheld its
    /// covariance (gam#2985): `PointCovarianceProvenance::explain`. Omitted when
    /// the point integrates the fit's own covariance.
    #[serde(skip_serializing_if = "Option::is_none")]
    point_covariance_note: Option<String>,
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
    /// so every payload carries the full fitted link identity back into the
    /// response-scale transforms (issue #1133).
    link_spec: String,
    /// The sampler that produced the draws, stamped by that sampler itself
    /// (`PosteriorSampler::label`): `"nuts"`, `"polya-gamma"`,
    /// `"polya-gamma-jeffreys"`, `"laplace"`, `"truncated-laplace"`, or
    /// `"conjugate-gaussian"`. Callers use it to badge the posterior or to warn
    /// when a class has fallen back to the approximate path.
    method: String,
    /// Metropolis acceptance rate of the draws (`PosteriorSampler::acceptance_rate`),
    /// present only for a sampler with an accept/reject step.
    acceptance_rate: Option<f64>,
    /// Whether `method` targets the model's exact posterior (the MCMC routes and
    /// the closed-form conjugate Gaussian route) rather than a Gaussian
    /// approximation of it (every Laplace form).
    exact: bool,
    /// Which coefficient covariance the draws describe, in the same
    /// vocabulary `predict()` reports (`"conditional"` or
    /// `"smoothing-corrected"`).
    covariance_source: String,
}

#[derive(Serialize)]
struct SampleConfigPayload {
    n_samples: usize,
    n_warmup: usize,
    n_chains: usize,
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
    /// Exact coefficient-covariance definition behind `survival_se`/`eta_se`
    /// (`"conditional"` or `"smoothing-corrected"`), owned by the engine
    /// result — never an echo of the request (#2296). `None` when no
    /// uncertainty was computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_source: Option<String>,
    /// Restriction horizon of the `rmst` column, in time units. Travels with
    /// the column because an RMST without its horizon is not interpretable —
    /// the same curve gives a different number at every `tau`. `None` exactly
    /// when the grid supports no horizon and `rmst` is therefore absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    rmst_tau: Option<f64>,
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
    #[serde(default)]
    covariance_source: Option<String>,
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
    // The version is shared by every commit between releases, so the commit
    // and the saved-model payload version are what tell two engines apart
    // (gam#3007, gam#3157). Both identity keys are None for a build that had no
    // gam git tree to read.
    info.set_item("commit", gam_build_identity::COMMIT)?;
    info.set_item("dirty", gam_build_identity::DIRTY)?;
    info.set_item(
        "model_payload_version",
        gam::inference::model::MODEL_PAYLOAD_VERSION,
    )?;
    info.set_item(
        "capabilities",
        vec![
            "fit",
            "fit_array",
            "load",
            "extend_with_group",
            "torch_from_fitted",
            "predict",
            "transformation_score",
            "latent_conditional_residual",
            "predict_array",
            "predict_conformal",
            "build_predict_payload_json",
            "interpolate_rows",
            "survival_chunk_defaults",
            "survival_chunk_ranges",
            "survival_should_chunk",
            "survival_chunk_iter_collect",
            "write_survival_csv",
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
#[pyclass(module = "gamfit._rust", name = "_EncodedTable", frozen, skip_from_py_object)]
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
                    // Legacy table entry points still infer from strings. Mark
                    // this one lazily-rendered row so numeric-looking labels
                    // retain their categorical source intent.
                    ColumnKindTag::Categorical => schema.present_cell_label(value, row).map(
                        |label| format!("{}{label}", gam::data::CATEGORICAL_CELL_SENTINEL),
                    ),
                    // A missing binary cell is refused, not rendered as "1".
                    ColumnKindTag::Binary => schema.present_cell_label(value, row),
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

/// One table column crossing the Python boundary, in the layout its source
/// declared. The Python adapter only reads that declaration (a NumPy dtype, a
/// categorical dtype); every decision about what the values mean is made in
/// gam-data.
#[derive(FromPyObject)]
enum PyTableColumn<'py> {
    /// A numeric vector, already `float64`.
    Numeric(PyReadonlyArray1<'py, f64>),
    /// A declared categorical: codes (`-1` = missing) into the level values.
    Categorical(PyReadonlyArray1<'py, i64>, Vec<Bound<'py, PyAny>>),
    /// A sequence of Python values with no declared type.
    Untyped(Bound<'py, PyAny>),
}

fn py_display_text(value: &Bound<'_, PyAny>) -> Result<String, gam::data::DataError> {
    value
        .str()
        .and_then(|text| text.to_str().map(str::to_owned))
        .map_err(|error| gam::data::DataError::InvalidValue {
            reason: format!("could not render a value as text: {error}"),
        })
}

/// Classify one Python value for [`gam::data::encode_untyped_column`]: text,
/// `None`, anything `float()` accepts (NaN is missing), or unsupported.
fn untyped_cell<'a>(value: &'a Bound<'_, PyAny>) -> gam::data::UntypedCell<'a> {
    use gam::data::UntypedCell;
    use pyo3::types::PyString;

    if let Ok(text) = value.cast::<PyString>() {
        return match text.to_str() {
            Ok(text) => UntypedCell::Text(text),
            Err(_) => UntypedCell::Unsupported("str"),
        };
    }
    if value.is_none() {
        return UntypedCell::Missing;
    }
    match value.extract::<f64>() {
        Ok(number) if number.is_nan() => UntypedCell::Missing,
        Ok(number) => UntypedCell::Number(number),
        Err(_) => UntypedCell::Unsupported(""),
    }
}

fn encode_py_table_column(
    name: &str,
    column: &PyTableColumn<'_>,
) -> Result<(SchemaColumn, Vec<f64>), gam::data::DataError> {
    match column {
        PyTableColumn::Numeric(values) => {
            let values = values.as_array().to_vec();
            let kind = gam::data::infer_numeric_column_kind(values.iter().copied());
            Ok((
                SchemaColumn {
                    name: name.to_string(),
                    kind,
                    levels: Vec::new(),
                },
                values,
            ))
        }
        PyTableColumn::Categorical(codes, levels) => {
            let codes = codes.as_array().to_vec();
            let levels = levels
                .iter()
                .map(py_display_text)
                .collect::<Result<Vec<_>, _>>()?;
            gam::data::encode_categorical_codes(name, &codes, &levels)
        }
        PyTableColumn::Untyped(source) => {
            let objects = source
                .try_iter()
                .and_then(|values| values.collect::<PyResult<Vec<_>>>())
                .map_err(|error| gam::data::DataError::InvalidValue {
                    reason: format!("column '{name}' is not a sequence of values: {error}"),
                })?;
            // Only the first unsupported cell is reported, so only its type is named.
            let unsupported_type: String;
            let mut cells = objects.iter().map(untyped_cell).collect::<Vec<_>>();
            if let Some(row) = cells
                .iter()
                .position(|cell| matches!(cell, gam::data::UntypedCell::Unsupported(_)))
            {
                unsupported_type = objects[row]
                    .get_type()
                    .fully_qualified_name()
                    .and_then(|type_name| type_name.to_str().map(str::to_owned))
                    .map_err(|error| gam::data::DataError::InvalidValue {
                        reason: format!(
                            "could not name the type of the value at row {}, column '{name}': {error}",
                            row + 1
                        ),
                    })?;
                cells[row] = gam::data::UntypedCell::Unsupported(&unsupported_type);
            }
            gam::data::encode_untyped_column(name, &cells, |row| py_display_text(&objects[row]))
        }
    }
}

/// Construct an encoded table from columns in their declared layouts: `float64`
/// vectors and categorical codes cross through rust-numpy without Python
/// objects, and untyped columns are classified cell by cell here and encoded by
/// gam-data's single untyped-column rule.
#[pyfunction]
fn encoded_table_from_columns(
    headers: Vec<String>,
    columns: Vec<PyTableColumn<'_>>,
) -> PyResult<PyEncodedTable> {
    ensure_unique_headers(&headers).map_err(py_value_error)?;
    if columns.len() != headers.len() {
        return Err(py_value_error(format!(
            "received {} columns for {} headers",
            columns.len(),
            headers.len()
        )));
    }
    let encoded = headers
        .iter()
        .zip(&columns)
        .map(|(name, column)| encode_py_table_column(name, column))
        .collect::<Result<Vec<_>, _>>()
        .map_err(data_error_to_pyerr)?;
    let n_rows = encoded.first().map_or(0, |(_, values)| values.len());
    if n_rows == 0 {
        return Err(data_error_to_pyerr(gam::data::DataError::EmptyInput {
            reason: "table data cannot be empty".to_string(),
        }));
    }
    let mut values = Array2::<f64>::zeros((n_rows, headers.len()));
    let mut schema_columns = Vec::with_capacity(headers.len());
    let mut column_kinds = Vec::with_capacity(headers.len());
    for (column, (schema, encoded_values)) in encoded.into_iter().enumerate() {
        if encoded_values.len() != n_rows {
            return Err(data_error_to_pyerr(gam::data::DataError::SchemaMismatch {
                reason: format!(
                    "column '{}' has {} rows but expected {n_rows}",
                    schema.name,
                    encoded_values.len()
                ),
            }));
        }
        values
            .column_mut(column)
            .assign(&ndarray::ArrayView1::from(&encoded_values));
        column_kinds.push(schema.kind);
        schema_columns.push(schema);
    }
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

/// Import a Polars/PyArrow provider through the Arrow C Stream
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
    let mut reader = ArrowArrayStreamReader::try_new(stream).map_err(|error| {
        data_error_to_pyerr(gam::data::DataError::ParseError {
            reason: format!("failed to import Arrow C stream: {error}"),
        })
    })?;
    let dataset =
        gam::data::encode_arrow_record_batch_reader_with_inferred_schema(&mut reader, headers)
            .map_err(data_error_to_pyerr)?;
    Ok(PyEncodedTable { dataset })
}

/// Project a typed prediction table onto the model's input columns and re-encode
/// it against the saved training schema.
///
/// The column set is the model's input contract ([`prediction_consumable_columns`]),
/// and a required column the table lacks is refused. The cell rules are gam-data's
/// [`gam::data::project_encoded_to_schema`], the projection `gam predict` applies
/// to a Parquet file: labels map onto training levels, a random-effect group's
/// unseen or missing label takes the unknown-level code, and a missing or
/// non-finite numeric cell is refused naming its column. A numeric-coded fixed
/// `factor(g)` has no categorical schema, so its unseen levels are refused by
/// [`FittedModel::unseen_numeric_factor_levels`], the check `gam predict` runs.
///
/// A refused cell is [`PredictError::Input`] and a missing column or a column
/// of the wrong kind is [`PredictError::SchemaMismatch`], so each reaches Python
/// as its own class.
fn dataset_with_model_schema_from_encoded(
    model: &FittedModel,
    source: &EncodedDataset,
) -> Result<EncodedDataset, PredictError> {
    let required = required_prediction_columns(model)?;
    let present = source.headers.iter().cloned().collect::<BTreeSet<_>>();
    let missing = required
        .difference(&present)
        .map(|name| format!("missing required column '{name}'"))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(PredictError::SchemaMismatch(missing.join(" ")));
    }
    let consumable = prediction_consumable_columns(model)?;
    let keep = source
        .headers
        .iter()
        .enumerate()
        .filter(|(_, name)| consumable.contains(name.as_str()))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let selected = EncodedDataset {
        headers: keep
            .iter()
            .map(|&index| source.headers[index].clone())
            .collect(),
        values: source.values.select(ndarray::Axis(1), &keep),
        schema: DataSchema {
            columns: keep
                .iter()
                .map(|&index| {
                    source.schema.columns.get(index).cloned().ok_or_else(|| {
                        format!(
                            "encoded table column '{}' has no source schema",
                            source.headers[index]
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        },
        column_kinds: keep
            .iter()
            .map(|&index| source.column_kinds[index])
            .collect(),
    };
    let policy = gam::data::UnseenCategoryPolicy::encode_unknown_for_columns(
        model.random_effect_group_columns(),
    );
    let schema = model
        .require_data_schema()
        .map_err(|error| PredictError::Other(error.to_string()))?;
    let dataset = gam::data::project_encoded_to_schema(selected, schema, &policy).map_err(
        |error| match error {
            gam::data::DataError::InvalidCell { .. } => PredictError::Input(error),
            gam::data::DataError::SchemaMismatch { reason } => PredictError::SchemaMismatch(reason),
            other => PredictError::Other(other.to_string()),
        },
    )?;
    if let Some(unseen) = model
        .unseen_numeric_factor_levels(&dataset.headers, dataset.values.view())
        .into_iter()
        .next()
    {
        return Err(PredictError::Input(unseen));
    }
    Ok(dataset)
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
    let consumable = prediction_consumable_columns(model)?;
    for (column_index, column) in source.headers.iter().enumerate() {
        if !consumable.contains(column.as_str()) {
            continue;
        }
        if let Some(row) = source
            .values
            .column(column_index)
            .iter()
            .position(|value| !value.is_finite())
        {
            issues.push(SchemaIssue {
                kind: "non_finite".to_string(),
                message: format!(
                    "non-finite value at row {}, column '{column}'",
                    row + 1
                ),
                column: Some(column.clone()),
            });
        }
    }
    if issues.is_empty()
        && let Err(error) = dataset_with_model_schema_from_encoded(model, source)
    {
        let column = match &error {
            PredictError::Input(gam::data::DataError::InvalidCell { column, .. }) => {
                Some(column.clone())
            }
            PredictError::Input(_) | PredictError::SchemaMismatch(_) | PredictError::Other(_) => {
                None
            }
        };
        issues.push(SchemaIssue {
            kind: "schema_error".to_string(),
            message: String::from(error),
            column,
        });
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
fn marginal_slope_clip_probabilities<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(values
        .as_array()
        .mapv(|value| value.clamp(0.0, 1.0))
        .into_pyarray(py)
        .unbind())
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
    match payload.covariance_source {
        Some(source) => out.set_item("covariance_source", source)?,
        None => out.set_item("covariance_source", py.None())?,
    }

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
    match object
        .get("covariance_source")
        .and_then(serde_json::Value::as_str)
    {
        Some(source) => out.set_item("covariance_source", source)?,
        None => out.set_item("covariance_source", py.None())?,
    }
    match object
        .get("interval_level")
        .and_then(serde_json::Value::as_f64)
    {
        Some(level) => out.set_item("interval_level", level)?,
        None => out.set_item("interval_level", py.None())?,
    }
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
    set_optional_competing_risks_matrix(py, &out, "hazard_se", object.get("hazard_se"))?;
    set_optional_competing_risks_matrix(py, &out, "hazard_lower", object.get("hazard_lower"))?;
    set_optional_competing_risks_matrix(py, &out, "hazard_upper", object.get("hazard_upper"))?;
    set_optional_competing_risks_matrix(py, &out, "survival", object.get("survival"))?;
    set_optional_competing_risks_matrix(py, &out, "survival_se", object.get("survival_se"))?;
    set_optional_competing_risks_matrix(py, &out, "survival_lower", object.get("survival_lower"))?;
    set_optional_competing_risks_matrix(py, &out, "survival_upper", object.get("survival_upper"))?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "cumulative_hazard",
        object.get("cumulative_hazard"),
    )?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "cumulative_hazard_se",
        object.get("cumulative_hazard_se"),
    )?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "cumulative_hazard_lower",
        object.get("cumulative_hazard_lower"),
    )?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "cumulative_hazard_upper",
        object.get("cumulative_hazard_upper"),
    )?;
    set_optional_competing_risks_matrix(py, &out, "cif", object.get("cif"))?;
    set_optional_competing_risks_matrix(py, &out, "cif_se", object.get("cif_se"))?;
    set_optional_competing_risks_matrix(py, &out, "cif_lower", object.get("cif_lower"))?;
    set_optional_competing_risks_matrix(py, &out, "cif_upper", object.get("cif_upper"))?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "overall_survival",
        object.get("overall_survival"),
    )?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "overall_survival_se",
        object.get("overall_survival_se"),
    )?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "overall_survival_lower",
        object.get("overall_survival_lower"),
    )?;
    set_optional_competing_risks_matrix(
        py,
        &out,
        "overall_survival_upper",
        object.get("overall_survival_upper"),
    )?;
    set_optional_competing_risks_vector(
        py,
        &out,
        "linear_predictor",
        object.get("linear_predictor"),
    )?;
    set_optional_competing_risks_vector(py, &out, "eta_se", object.get("eta_se"))?;
    set_optional_competing_risks_vector(py, &out, "eta_lower", object.get("eta_lower"))?;
    set_optional_competing_risks_vector(py, &out, "eta_upper", object.get("eta_upper"))?;

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
    // Every prediction row needs its own id: a missing id cell is refused, not
    // relabelled as the first level (`NaN as usize == 0`) or as binary "1".
    let row_ids = (0..rows.dataset.values.nrows())
        .map(|row| {
            schema
                .present_cell_label(rows.dataset.values[[row, index]], row)
                .map_err(|err| py_value_error(format!("id_column '{id_column}': {err}")))
        })
        .collect::<PyResult<Vec<String>>>()?;
    Ok(Some(row_ids))
}

fn default_survival_time_grid_from_model(
    model: &FittedModel,
    dataset: &EncodedDataset,
) -> Result<Option<Vec<f64>>, String> {
    if !matches!(model.predict_model_class(), PredictModelClass::Survival) {
        return Ok(None);
    }
    let training_hi =
        gam::families::survival::predict::survival_training_time_upper_bound(model.payload());
    gam::families::survival::predict::default_survival_time_grid(
        model.payload().formula.as_str(),
        dataset,
        training_hi,
    )
}

#[pyfunction(signature = (
    headers,
    rows,
    formula,
    config_json = None,
    fisher_rao_w = None,
    warm_start_model = None
))]
fn fit_table(
    py: Python<'_>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    formula: String,
    config_json: Option<String>,
    fisher_rao_w: Option<PyReadonlyArray3<'_, f64>>,
    warm_start_model: Option<Vec<u8>>,
) -> PyResult<Py<PyBytes>> {
    // PyO3 0.28 names the old `allow_threads` API `detach`: the closure
    // runs without the GIL, so Python signal handling (KeyboardInterrupt,
    // SIGALRM handlers, etc.) can run while the Rust solver is in progress.
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    // The multinomial-logit family is a vector-response fit with its own
    // driver and persistence envelope; route it here on the same predicate the
    // CLI uses, so callers read the model kind off the returned bytes
    // (`saved_model_kind`) instead of re-deriving it from the family name.
    // A refused configuration is an `InvalidConfigurationError` here exactly as
    // it is once the fit runs (`fit_dataset_impl`), not a bare `GamfitError`.
    let fit_config = parse_fit_config(config_json.as_deref())
        .map_err(|reason| {
            workflow_error_to_pyerr(
                py,
                gam::families::fit_orchestration::WorkflowError::InvalidConfig { reason },
            )
        })?;
    if fit_config
        .family
        .as_deref()
        .is_some_and(gam::families::fit_orchestration::is_multinomial_family_name)
    {
        if warm_start_model.is_some() {
            return Err(py_value_error(
                "warm_start_from is not supported for multinomial fits".to_string(),
            ));
        }
        if fisher_rao_w.is_some() {
            return Err(py_value_error(
                "fisher_rao_w is not supported for multinomial fits".to_string(),
            ));
        }
        let model_bytes = detach_pyresult(py, "fit_multinomial", move || {
            fit_multinomial_dataset(&dataset, &formula, &fit_config)
        })?;
        return Ok(PyBytes::new(py, &model_bytes).unbind());
    }
    let fisher_values = fisher_rao_w.as_ref().map(|w| w.as_array().to_owned());
    let model_bytes = detach_workflow_result(py, "fit_table", move || {
        fit_dataset_impl(
            dataset,
            formula,
            config_json.as_deref(),
            fisher_values.as_ref().map(|w| w.view()),
            warm_start_model.as_deref(),
        )
    })?;
    Ok(PyBytes::new(py, &model_bytes).unbind())
}

#[pyfunction(signature = (
    x,
    y,
    formula,
    config_json = None,
    fisher_rao_w = None,
    warm_start_model = None
))]
fn fit_array(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    formula: String,
    config_json: Option<String>,
    fisher_rao_w: Option<PyReadonlyArray3<'_, f64>>,
    warm_start_model: Option<Vec<u8>>,
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
            warm_start_model.as_deref(),
        )
    })?;
    Ok(PyBytes::new(py, &model_bytes).unbind())
}

#[pyfunction]
fn compile_model(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<PyFittedModel> {
    PyFittedModel::compile(py, model_bytes)
}

/// Akaike evidence ratio of model A over model B on the smoothing-corrected
/// AIC, `exp(−(AIC_c(A) − AIC_c(B))/2)`, formed by the same Rust comparison as
/// `compare_models`; `+inf` / `0` once the ratio leaves the `f64` range.
///
/// This is the relative likelihood of Burnham & Anderson, NOT a Bayes factor
/// (no prior is integrated over), and the Python surface names it
/// `Model.evidence_ratio_vs` accordingly.
#[pyfunction]
fn evidence_ratio(
    py: Python<'_>,
    model_a: PyRef<'_, PyFittedModel>,
    model_b: PyRef<'_, PyFittedModel>,
) -> PyResult<f64> {
    let model_a = Arc::clone(&model_a.model);
    let model_b = Arc::clone(&model_b.model);
    detach_py_result(py, "evidence_ratio", move || {
        saved_models_evidence_ratio(&model_a, &model_b)
    })
}

/// The LAML-estimated `(σ, ν)` of a scaled Student-t fit, read off the compiled
/// model's likelihood; `None` for every other response family.
#[pyfunction]
fn student_t_parameters_from_model(model: PyRef<'_, PyFittedModel>) -> Option<(f64, f64)> {
    match model_likelihood_spec(&model.model).response {
        ResponseFamily::StudentT { sigma, nu } => Some((sigma, nu)),
        _ => None,
    }
}

/// Schema tag of the response-geometry saved-model container (#2114).
pub(crate) const RESPONSE_GEOMETRY_SCHEMA: &str = "gamfit.ResponseGeometryModel/v1";

/// Whether `family` names the multinomial-logit family, by the one engine
/// predicate `fit_table` and the CLI route on. Python front ends that must
/// choose before fitting (the sklearn classifier's binary/multi-class split)
/// ask here instead of keeping their own spelling list.
#[pyfunction]
fn is_multinomial_family_name(family: &str) -> bool {
    gam::families::fit_orchestration::is_multinomial_family_name(family)
}

/// The kind of a saved gamfit model payload, read from its JSON header. A
/// `gamfit.ManifoldSAE` schema of any version is `"manifold_sae"`, so a stale
/// version reaches its own refusal; the response-geometry container is
/// `"response_geometry"`; the multinomial envelope is `"multinomial"`. Every
/// other payload, including bytes that are not JSON, is `"scalar"`, whose
/// loader reports what is wrong with it.
#[pyfunction]
fn saved_model_kind(model_bytes: Vec<u8>) -> &'static str {
    #[derive(Deserialize)]
    struct SavedModelHeader {
        #[serde(default)]
        schema: Option<String>,
        #[serde(default)]
        model_class: Option<String>,
    }
    let Ok(header) = serde_json::from_slice::<SavedModelHeader>(&model_bytes) else {
        return "scalar";
    };
    let schema_family = header
        .schema
        .as_deref()
        .and_then(|schema| schema.split_once('/'))
        .map(|(family, _)| family);
    let manifold_family = crate::manifold::manifold_sae_payload::SCHEMA_TAG
        .split_once('/')
        .map(|(family, _)| family);
    if schema_family.is_some() && schema_family == manifold_family {
        return "manifold_sae";
    }
    if header.schema.as_deref() == Some(RESPONSE_GEOMETRY_SCHEMA) {
        return "response_geometry";
    }
    if header.model_class.as_deref() == Some(gam::families::multinomial::MULTINOMIAL_MODEL_CLASS) {
        return "multinomial";
    }
    "scalar"
}

/// Write a saved gamfit model's bytes to `path` through the one saved-model
/// writer every surface shares (gam#3054): atomic, so a failed save leaves the
/// previous file whole, and durable on Unix before it returns.
#[pyfunction]
fn write_saved_model_file(
    py: Python<'_>,
    path: std::path::PathBuf,
    model_bytes: Vec<u8>,
) -> PyResult<()> {
    py.detach(move || gam_model_api::saved_model::write_saved_model(&path, &model_bytes))
        .map_err(crate::ffi::ffi_errors::saved_document_error_to_pyerr)
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
    model: PyRef<'_, PyFittedModel>,
    request_json: String,
) -> PyResult<Py<PyBytes>> {
    let model = Arc::clone(&model.model);
    let out = detach_py_result(py, "extend_model_with_group", move || {
        extend_model_with_group_impl(&model, &request_json)
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
    };
    serde_json::to_string(&payload)
        .map_err(|err| py_value_error(format!("failed to serialize predict payload: {err}")))
}

#[pyfunction(signature = (model, headers, rows, interval, covariance_mode=None, observation_interval=None))]
fn build_model_predict_payload_json(
    model: PyRef<'_, PyFittedModel>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    interval: Option<f64>,
    covariance_mode: Option<String>,
    observation_interval: Option<bool>,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let time_grid = if matches!(
        model.model.model_kind,
        gam::inference::model::ModelKind::Survival
    ) {
        gam::families::survival::predict::default_survival_time_grid(
            model.model.payload().formula.as_str(),
            &rows.dataset,
            gam::families::survival::predict::survival_training_time_upper_bound(
                model.model.payload(),
            ),
        )
        .map_err(py_value_error)?
    } else {
        None
    };
    build_predict_payload_json(interval, time_grid, covariance_mode, observation_interval)
}

#[pyfunction(signature = (model, headers, rows, interval, covariance_mode=None, observation_interval=None))]
fn predict_table(
    py: Python<'_>,
    model: PyRef<'_, PyFittedModel>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    interval: Option<f64>,
    covariance_mode: Option<String>,
    observation_interval: Option<bool>,
) -> PyResult<PyObject> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let model = Arc::clone(&model.model);
    detach_predict_result(py, "predict_table", move || {
        predict_encoded_table_configured_impl(
            &model,
            dataset,
            interval,
            covariance_mode,
            observation_interval,
        )
    })?
    .into_py(py)
}

#[pyfunction]
fn ctn_required_fit_columns(formula: String, config_json: String) -> PyResult<Vec<String>> {
    let config = parse_fit_config(Some(&config_json)).map_err(py_value_error)?;
    gam::inference::ctn::required_fit_columns(&formula, &config)
        .map(|columns| columns.into_iter().collect()).map_err(py_value_error)
}

#[pyfunction]
fn required_model_columns(model: PyRef<'_, PyFittedModel>, observed_score: bool) -> PyResult<Option<Vec<String>>> {
    let model = model.model.as_ref();
    // Outcome models without an embedded CTN retain their existing table
    // ingestion contract, including intercept-only row-count inputs.
    if !observed_score && model.score_transform.is_none() {
        return Ok(None);
    }
    let transformed;
    let model = match model.score_transform.as_ref() {
        Some(transform) if observed_score => {
            transformed = FittedModel::from_payload((**transform).clone());
            &transformed
        }
        _ => model,
    };
    let mut columns = model.prediction_required_columns().map_err(py_value_error)?;
    if observed_score {
        let response = response_column_name(&model.formula)
            .ok_or_else(|| py_value_error("CTN requires a named observed response".to_string()))?;
        columns.insert(response);
    }
    Ok(Some(columns.into_iter().collect()))
}

/// Column names a positional (NumPy) prediction array of `width` columns
/// binds to.
///
/// A model fitted from a positional array reads the synthetic sequence
/// `x0..x{width-1}`, so that sequence is used whenever it covers every column
/// the model reads. A model fitted from a named table binds the array to its
/// predictor columns in training-table order (the order the sklearn wrapper
/// reports as `feature_names_in_`), and only when the width equals their
/// count; any other width is a `SchemaMismatchError` naming the expected
/// columns, never a guessed binding.
#[pyfunction]
fn positional_prediction_headers(
    model: PyRef<'_, PyFittedModel>,
    width: usize,
) -> PyResult<Vec<String>> {
    let model = model.model.as_ref();
    let required = model.prediction_required_columns().map_err(py_value_error)?;
    let synthetic: Vec<String> = (0..width).map(|index| format!("x{index}")).collect();
    if required.iter().all(|name| synthetic.contains(name)) {
        return Ok(synthetic);
    }
    let predictors: Vec<String> = model
        .payload()
        .training_headers
        .iter()
        .flatten()
        .filter(|name| required.contains(name.as_str()))
        .cloned()
        .collect();
    if predictors.len() == required.len() && predictors.len() == width {
        return Ok(predictors);
    }
    Err(SchemaMismatchError::new_err(format!(
        "a positional array binds to the model's {} predictor column(s) {:?} in \
         training-table order, but the input has {width} column(s); pass a table \
         with named columns or an array of matching width",
        required.len(),
        if predictors.len() == required.len() {
            predictors
        } else {
            required.into_iter().collect()
        },
    )))
}

fn transformation_score_encoded_table_impl(
    model: &FittedModel,
    source: EncodedDataset,
) -> Result<Array1<f64>, String> {
    let transformed;
    let model = match model.score_transform.as_ref() {
        Some(transform) => {
            transformed = FittedModel::from_payload((**transform).clone());
            &transformed
        }
        None => model,
    };
    if model.predict_model_class() != PredictModelClass::TransformationNormal {
        return Err(format!(
            "transformation_score requires a transformation-normal model; got '{}'",
            prediction_model_class_label(&model)
        ));
    }
    let response_name =
        response_column_name(model.payload().formula.as_str()).ok_or_else(|| {
            "transformation-normal model formula has no plain observed-response column".to_string()
        })?;
    if !source.headers.iter().any(|header| header == &response_name) {
        return Err(format!(
            "transformation_score data must contain observed response column '{response_name}'"
        ));
    }
    let dataset = dataset_with_model_schema_from_encoded(&model, &source)?;
    let col_map = dataset.column_map();
    let response_column = *col_map.get(&response_name).ok_or_else(|| {
        format!(
            "transformation_score projection dropped observed response column '{response_name}'"
        )
    })?;
    let response = dataset.values.column(response_column).to_owned();
    let offset = resolve_offset_column(&dataset, &col_map, model.offset_column.as_deref())?;
    build_transformation_normal_observed_scores(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &response,
        &offset,
    )
}

/// Evaluate the labelled-data CTM score `Phi^-1(F_hat(y_i | x_i))`.
/// Ordinary `predict_table` remains response-scale `E[Y|x]`; keeping these as
/// distinct typed entries prevents a conditional mean from being consumed as
/// a generated marginal-slope regressor.
#[pyfunction]
fn transformation_score_table<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyFittedModel>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
) -> PyResult<Py<PyArray1<f64>>> {
    let model = Arc::clone(&model.model);
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let scores = detach_py_result(py, "transformation_score_table", move || {
        transformation_score_encoded_table_impl(&model, dataset)
    })?;
    Ok(scores.into_pyarray(py).unbind())
}

/// The declared conditional latent law's standardized residual
/// `ζ = (z − m(a))/√v(a)` of a saved marginal-slope model on new rows, through
/// the map its fit applied (gam#3016). `None` when the fit consumed no
/// conditional law. The frame needs the score and the conditioning covariates,
/// not a survival model's time columns.
#[pyfunction]
fn latent_conditional_residual_table<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyFittedModel>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
) -> PyResult<Option<Py<PyArray1<f64>>>> {
    let model = Arc::clone(&model.model);
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let residual = detach_pyresult(py, "latent_conditional_residual_table", move || {
        let required = model
            .latent_conditional_residual_columns()
            .map_err(py_value_error)?;
        let present = dataset.headers.iter().cloned().collect::<BTreeSet<_>>();
        let missing = required
            .difference(&present)
            .map(|name| format!("missing required column '{name}'"))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(SchemaMismatchError::new_err(missing.join(" ")));
        }
        model
            .latent_conditional_residual(dataset.values.view(), &dataset.column_map())
            .map_err(|error| PredictInputError::new_err(error.to_string()))
    })?;
    Ok(residual.map(|values| values.into_pyarray(py).unbind()))
}

/// Per-row residuals of type `kind` (`response`, `working`, `deviance`,
/// `pearson`) of a saved standard model on labeled rows; the rows must carry
/// the response (and the weight/offset columns the model was fit with).
#[pyfunction]
fn residuals_table<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyFittedModel>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    kind: String,
) -> PyResult<Py<PyArray1<f64>>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let kind = kind
        .parse::<gam::solver::pirls::ResidualKind>()
        .map_err(py_value_error)?;
    let source = rows.dataset.clone();
    let model = Arc::clone(&model.model);
    let residuals = detach_py_result(py, "residuals_table", move || {
        let dataset = dataset_with_model_schema_from_encoded(&model, &source)?;
        gam::families::inference::saved_residuals::saved_model_residuals(&model, &dataset, kind)
    })?;
    Ok(residuals.into_pyarray(py).unbind())
}

/// Distribution-free conformal prediction intervals (issue #310 family path).
///
/// Runs the standard model-based predictor on `(headers, rows)`, then replaces
/// the response-scale `mean_lower` / `mean_upper` with the split-conformal
/// interval `μ̂(x) ± q̂·s(x)` calibrated at `conformal_level` from the held-out
/// `(calibration_headers, calibration_rows)` fold — which must contain the
/// response column. The returned interval carries finite-sample marginal
/// coverage `≥ conformal_level` regardless of model misspecification.
#[pyfunction(signature = (model, headers, rows, calibration_headers, calibration_rows, conformal_level, options_json=None))]
fn predict_table_conformal(
    py: Python<'_>,
    model: PyRef<'_, PyFittedModel>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    calibration_headers: Vec<String>,
    calibration_rows: PyRef<'_, PyEncodedTable>,
    conformal_level: f64,
    options_json: Option<String>,
) -> PyResult<PyObject> {
    let model = Arc::clone(&model.model);
    rows.require_headers(&headers).map_err(py_value_error)?;
    calibration_rows
        .require_headers(&calibration_headers)
        .map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let calibration_dataset = calibration_rows.dataset.clone();
    let payload = detach_py_result(py, "predict_table_conformal", move || {
        predict_encoded_table_conformal_impl(
            &model,
            dataset,
            calibration_dataset,
            conformal_level,
            options_json.as_deref(),
        )
    })?;
    prediction_payload_into_py(py, payload)
}

#[pyfunction]
fn predict_array<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyFittedModel>,
    x: PyReadonlyArray2<'py, f64>,
    options_json: Option<String>,
) -> PyResult<Py<PyArray2<f64>>> {
    let model = Arc::clone(&model.model);
    let x_values = x.as_array().to_owned();
    let out = detach_py_result(py, "predict_array", move || {
        predict_array_impl(&model, x_values.view(), options_json.as_deref())
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
    // Endpoints whose hazard grids differ in shape cannot be stacked.
    let cumulative_hazard =
        ndarray::stack(Axis(0), &endpoint_views).map_err(shape_error_to_pyerr)?;
    // Typed engine path: `assemble_competing_risks_cif` returns
    // `Result<_, SurvivalError>`, raised as the class of its fit category.
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
    // Typed engine path: `SurvivalError` → the class of its fit category
    // (issue #343), no string flattening.
    let result = gam::families::survival::assemble_competing_risks_cif_from_endpoints(
        times,
        cumulative_hazards,
    )
    .map_err(survival_error_to_pyerr)?;
    let cif = result.cif;
    Ok((cif, result.overall_survival))
}

#[pyfunction]
fn build_sample_payload_json(samples: Option<i64>, seed: Option<i64>) -> PyResult<String> {
    let mut payload = serde_json::Map::new();
    if let Some(value) = samples {
        payload.insert("samples".to_string(), serde_json::Value::from(value));
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
    model: PyRef<'_, PyFittedModel>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    options_json: Option<String>,
) -> PyResult<Py<PyDict>> {
    let model = Arc::clone(&model.model);
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let payload = detach_py_result(py, "sample_table", move || {
        sample_encoded_table_impl(&model, dataset, options_json.as_deref())
    })?;
    let config = PyDict::new(py);
    config.set_item("n_samples", payload.config.n_samples)?;
    config.set_item("n_warmup", payload.config.n_warmup)?;
    config.set_item("n_chains", payload.config.n_chains)?;
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
    out.set_item("acceptance_rate", payload.acceptance_rate)?;
    out.set_item("exact", payload.exact)?;
    out.set_item("covariance_source", payload.covariance_source)?;
    Ok(out.unbind())
}

// paired_sample_table and paired_cumulative_incidence_table pyffi wrappers
// removed: their payload types (PairedSamplePayload, PairedCifPayload,
// PyPairedCifOptions) were never defined in the main crate; the orphaned
// implementations and their helpers were deleted below.

fn dense_affine_design_to_python(
    py: Python<'_>,
    affine: DenseAffineDesign,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("offset", affine.offset.into_pyarray(py))?;
    let matrix = affine.matrix.into_pyarray(py);
    match affine.eta_gradient {
        Some(gradient) => out.set_item("eta_gradient", gradient.into_pyarray(py))?,
        // The fitted predictor is linear in its coefficients here, so the value
        // operator IS ∂η/∂β. Bind the same array under both names instead of
        // duplicating an n x p buffer.
        None => out.set_item("eta_gradient", &matrix)?,
    }
    out.set_item("matrix", &matrix)?;
    out.set_item("coefficients", affine.coefficients.into_pyarray(py))?;
    out.set_item("coefficient_frame", affine.coefficient_frame)?;
    out.set_item("coefficient_start", affine.coefficient_start)?;
    out.set_item("coefficient_stop", affine.coefficient_stop)?;
    match affine.covariance_conditional {
        Some(covariance) => out.set_item("covariance_conditional", covariance.into_pyarray(py))?,
        None => out.set_item("covariance_conditional", py.None())?,
    }
    match affine.covariance_smoothing_corrected {
        Some(covariance) => out.set_item(
            "covariance_smoothing_corrected",
            covariance.into_pyarray(py),
        )?,
        None => out.set_item("covariance_smoothing_corrected", py.None())?,
    }
    match affine.covariance_frequentist {
        Some(covariance) => out.set_item("covariance_frequentist", covariance.into_pyarray(py))?,
        None => out.set_item("covariance_frequentist", py.None())?,
    }
    Ok(out.unbind())
}

#[pyfunction]
fn affine_design_table(
    py: Python<'_>,
    model: PyRef<'_, PyFittedModel>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
) -> PyResult<Py<PyDict>> {
    let model = Arc::clone(&model.model);
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let affine = detach_py_result(py, "affine_design_table", move || {
        affine_design_encoded_table_impl(&model, dataset)
    })?;
    dense_affine_design_to_python(py, affine)
}

#[pyfunction]
fn affine_design_array<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyFittedModel>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let model = Arc::clone(&model.model);
    let x_values = x.as_array().to_owned();
    let affine = detach_py_result(py, "affine_design_array", move || {
        affine_design_array_impl(&model, x_values.view())
    })?;
    dense_affine_design_to_python(py, affine)
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

/// Return the terms-layer orthonormal coefficient chart for the weighted
/// sum-to-zero constraint `1ᵀ W B Z = 0`.
///
/// Python performs the `B @ Z` and `Z.T @ S @ Z` products in torch so the fit
/// remains connected to differentiable design/penalty inputs; Rust remains
/// the sole owner of the rank convention and gauge construction.
#[pyfunction(signature = (basis, weights = None))]
fn weighted_sum_to_zero_transform<'py>(
    py: Python<'py>,
    basis: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let basis_view = basis.as_array();
    if basis_view.nrows() == 0 {
        return Err(py_value_error(
            "weighted_sum_to_zero_transform requires at least one basis row".to_string(),
        ));
    }
    if let Some(((row, col), value)) = basis_view
        .indexed_iter()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(py_value_error(format!(
            "basis[{row},{col}] must be finite; got {value}"
        )));
    }
    let weights_view = match weights.as_ref() {
        Some(weights) => {
            let view = weights.as_array();
            if view.len() != basis_view.nrows() {
                return Err(py_value_error(format!(
                    "weights length {} does not match basis rows {}",
                    view.len(),
                    basis_view.nrows()
                )));
            }
            if let Some((row, value)) = view
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite() || **value < 0.0)
            {
                return Err(py_value_error(format!(
                    "weights[{row}] must be finite and non-negative; got {value}"
                )));
            }
            if view.iter().all(|value| *value == 0.0) {
                return Err(py_value_error(
                    "weighted_sum_to_zero_transform requires positive total weight".to_string(),
                ));
            }
            Some(view)
        }
        None => None,
    };
    let (_, transform) = gam::terms::basis::apply_sum_to_zero_constraint(basis_view, weights_view)
        .map_err(basis_error_to_pyerr)?;
    Ok(transform.into_pyarray(py).unbind())
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

    let requested_nullspace = duchon_nullspace_order_from_m(m);
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
    let penalty = built
        .active_penalties
        .iter()
        .find(|penalty| {
            matches!(
                penalty.info.source,
                gam::terms::basis::PenaltySource::Primary
            )
        })
        .ok_or_else(|| {
            py_value_error("duchon_basis_with_jet: primary penalty was not built".to_string())
        })?
        .matrix
        .clone();

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
        length_scale: MaternLengthScale::fixed(length_scale),
        nu: nu_parsed,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::None,
        aniso_log_scales: aniso_vec,
        periodic: None,
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
        "sphere" => ambient_sphere_basis_with_jet(py, t, 2),
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
                    // `n_basis` means the same thing in both branches: the
                    // number of design columns on the unit parameter domain.
                    // A periodic basis takes its knots as the uniform lattice
                    // `linspace(0, 1, n_basis + 1)` (one cyclic control per
                    // interval, see `periodic_knot_domain`); an open basis
                    // takes the canonical clamped uniform vector with
                    // `n_basis - (degree + 1)` internal knots.
                    if periodic {
                        Array1::linspace(0.0, 1.0, n_basis + 1)
                    } else {
                        gam::terms::basis::generate_full_knot_vector(
                            (0.0, 1.0),
                            n_basis - (degree + 1),
                            degree,
                        )
                        .map_err(basis_error_to_pyerr)?
                    }
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
            // the closed form `phi'` that the latent periodic-curve fits
            // also rely on. The non-periodic branch uses the open-uniform
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
                if null_basis.ncols() > penalty.ncols() {
                    return Err(py_value_error(format!(
                        "basis_with_jet bspline returned a nullspace with {} columns for a {}-coefficient penalty",
                        null_basis.ncols(),
                        penalty.ncols()
                    )));
                }
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
    // Spec chart, NOT the data-metric one (gam#237). This primitive returns a
    // design matrix and nothing else — there is no fit here to freeze a chart
    // into and replay at predict time, so adopting a chart derived from the
    // realized Gram made the COLUMN COUNT a function of the ROW count:
    // `cols = min(K, n_points + d + 1)`, measured — the same 12-center d=2 spec
    // gave 4 columns on 1 row and 12 on 30. Deriving the chart from the centers
    // makes the width `K` for every frame size (63 of 63 configurations).
    let built = build_duchon_basis_spec_chart(pts, &spec).map_err(basis_error_to_pyerr)?;
    Ok(built.design.to_dense().into_pyarray(py).unbind())
}

/// The knots or centers a 1-D basis-evaluation helper builds on `t`:
/// `(locations, effective_order, shrunk)`. `knots_or_centers` is `None` (the
/// formula front door's default for the kind on `t`), an integer size, or an
/// explicit float64 vector; [`resolve_basis_locations_1d`] owns all three.
#[pyfunction(signature = (t, basis_kind, knots_or_centers = None, order = 3, periodic = false))]
fn resolve_basis_locations_1d<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    basis_kind: &str,
    knots_or_centers: Option<&Bound<'py, PyAny>>,
    order: usize,
    periodic: bool,
) -> PyResult<(Py<PyArray1<f64>>, usize, bool)> {
    let kind = PositionBasisKind::parse(basis_kind).map_err(py_value_error)?;
    let request = position_basis_locations_arg(knots_or_centers)?;
    let resolved =
        gam::terms::basis::position_basis::resolve_basis_locations_1d(
            t.as_array(),
            kind,
            request,
            order,
            periodic,
        )
        .map_err(py_value_error)?;
    Ok((
        resolved.locations.into_pyarray(py).unbind(),
        resolved.order,
        resolved.shrunk,
    ))
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
        duchon_nullspace_order_from_m(m),
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
    let penalty = core_duchon_function_norm_penalty(
        center_matrix.view(),
        spec_length_scale,
        spec_nullspace,
        spec_power,
        &periodic_flags,
        period,
    )
    .map_err(basis_error_to_pyerr)?;
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
        "harmonic" => (
            SphereMethod::Harmonic,
            SphereWahbaKernel::Sobolev,
            Some(n_centers),
        ),
        other => {
            return Err(py_value_error(format!(
                "sphere_basis kernel must be one of 'sobolev', 'harmonic'; got '{other}'"
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
        adaptive_degree: false,
    };
    let built = build_spherical_spline_basis(pts, &spec).map_err(basis_error_to_pyerr)?;
    let penalty = built
        .active_penalties
        .iter()
        .find(|penalty| {
            matches!(
                penalty.info.source,
                gam::terms::basis::PenaltySource::Primary
            )
        })
        .ok_or_else(|| {
            py_value_error("sphere_basis: primary penalty was not built; check spec".to_string())
        })?
        .matrix
        .clone();
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
        "harmonic" => (
            SphereMethod::Harmonic,
            SphereWahbaKernel::Sobolev,
            Some(ctrs.nrows()),
        ),
        other => {
            return Err(py_value_error(format!(
                "sphere_basis_with_centers kernel must be one of 'sobolev', 'harmonic'; got '{other}'"
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
        adaptive_degree: false,
    };
    let built = build_spherical_spline_basis(pts, &spec).map_err(basis_error_to_pyerr)?;
    let penalty = built
        .active_penalties
        .iter()
        .find(|penalty| {
            matches!(
                penalty.info.source,
                gam::terms::basis::PenaltySource::Primary
            )
        })
        .ok_or_else(|| {
            py_value_error(
                "sphere_basis_with_centers: primary penalty was not built; check spec".to_string(),
            )
        })?
        .matrix
        .clone();
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
        "harmonic" => Ok((SphereMethod::Harmonic, SphereWahbaKernel::Sobolev)),
        other => Err(py_value_error(format!(
            "{site} kernel must be one of 'sobolev', 'harmonic'; got '{other}'"
        ))),
    }
}

/// Column count of the S² basis `sphere_basis` / `sphere_basis_with_centers`
/// build for `n_centers` centers (the harmonic truncation degree under
/// `kernel = "harmonic"`), read without evaluating it through
/// `gam::terms::basis::spherical_spline_basis_width`, the builder's own width
/// rule. A descriptor the builder refuses (a harmonic degree past the cap,
/// fewer than two Wahba centers) is refused here with the same error.
#[pyfunction(signature = (n_centers, kernel = "sobolev"))]
fn sphere_basis_size(n_centers: usize, kernel: &str) -> PyResult<usize> {
    let (method, wahba_kernel) = sphere_kernel_kind_from_str(kernel, "sphere_basis_size")?;
    let max_degree = matches!(method, SphereMethod::Harmonic).then_some(n_centers);
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: n_centers,
        },
        penalty_order: 2,
        double_penalty: false,
        radians: false,
        method,
        max_degree,
        wahba_kernel,
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
        adaptive_degree: false,
    };
    gam::terms::basis::spherical_spline_basis_width(&spec, 0).map_err(basis_error_to_pyerr)
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
        adaptive_degree: false,
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
        adaptive_degree: false,
    };
    let jet = spherical_spline_design_jet(pts, &spec).map_err(basis_error_to_pyerr)?;
    Ok(jet.into_pyarray(py).unbind())
}

/// Analytic DESIGN hessian `∂²Φ/∂(lat, lon)²` of the spherical-spline basis,
/// shape `(N, K, 2, 2)` in the same angular units as the input. With `centers`
/// it mirrors `sphere_basis_jet_with_centers`, otherwise `sphere_basis_jet`
/// (auto farthest-point centers, or harmonic degree `L = n_centers`), column for
/// column. A Wahba basis refuses an evaluation point that coincides with a
/// center of a kernel that has no Hessian there.
#[pyfunction(signature = (points, n_centers, centers = None, penalty_order = 2, kernel = "sobolev", radians = false))]
fn sphere_basis_hessian<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    n_centers: usize,
    centers: Option<PyReadonlyArray2<'py, f64>>,
    penalty_order: usize,
    kernel: &str,
    radians: bool,
) -> PyResult<Py<PyArray4<f64>>> {
    let pts = points.as_array();
    if pts.ncols() != 2 {
        return Err(py_value_error(format!(
            "sphere_basis_hessian expects points of shape (N, 2) [lat, lon]; got d={}",
            pts.ncols()
        )));
    }
    if !(1..=4).contains(&penalty_order) {
        return Err(py_value_error(format!(
            "sphere_basis_hessian penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        )));
    }
    let (method, wahba_kernel) = sphere_kernel_kind_from_str(kernel, "sphere_basis_hessian")?;
    let harmonic = matches!(method, SphereMethod::Harmonic);
    let (center_strategy, max_degree) = match centers.as_ref() {
        Some(ctrs) => {
            let ctrs = ctrs.as_array();
            if ctrs.ncols() != 2 {
                return Err(py_value_error(format!(
                    "sphere_basis_hessian expects centers of shape (K, 2) [lat, lon]; got d={}",
                    ctrs.ncols()
                )));
            }
            (
                CenterStrategy::UserProvided(ctrs.to_owned()),
                harmonic.then_some(ctrs.nrows()),
            )
        }
        None => (
            CenterStrategy::FarthestPoint {
                num_centers: n_centers,
            },
            harmonic.then_some(n_centers),
        ),
    };
    let spec = SphericalSplineBasisSpec {
        center_strategy,
        penalty_order,
        double_penalty: false,
        radians,
        method,
        max_degree,
        wahba_kernel,
        identifiability: SphericalSplineIdentifiability::CenterSumToZero,
        adaptive_degree: false,
    };
    let hessian = spherical_spline_design_hessian(pts, &spec).map_err(basis_error_to_pyerr)?;
    Ok(hessian.into_pyarray(py).unbind())
}

/// Real spherical harmonics on `S²` in AMBIENT coordinates, with analytic jet.
///
/// `t` is an `(N, 3)` array of unit vectors `(x, y, z)`. The columns are the
/// `(degree+1)²` real harmonics through `degree`, and the returned penalty is
/// the exact Laplace-Beltrami spectrum `[l(l+1)]²` — derived, not tabulated.
///
/// This replaced a seven-column `(lat, lon)` chart helper. `S²` admits no
/// global 2-D chart, and that one paid for it: the poles were an optimiser
/// boundary, longitude was gauge there, the trust-region metric was wrong by
/// `cos²(lat)`, and the span was not closed under `SO(3)`. The ambient form has
/// none of those, and at degree 2 it spans all five `l = 2` harmonics where the
/// chart carried three.
#[pyfunction(signature = (t, degree = 2))]
fn ambient_sphere_basis_with_jet<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    degree: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>, Py<PyArray2<f64>>)> {
    // The evaluator and its spectrum live in the core SAE path; this helper only
    // routes the caller's coordinates through that single source of truth, which
    // is what keeps the core and PyFFI derivatives from drifting.
    let coords = t.as_array();
    let evaluator = AmbientSphereHarmonicEvaluator::new(degree).map_err(py_value_error)?;
    let (phi, jet) = evaluator.evaluate(coords).map_err(py_value_error)?;
    let modes = evaluator.spectral_modes();
    let mut penalty = Array2::<f64>::zeros((modes.len(), modes.len()));
    for (column, mode) in modes.iter().enumerate() {
        penalty[[column, column]] = mode.l2_gram_weight * mode.laplace_eigenvalue.powi(2);
    }
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
    // Estimand-explicit schema (#2785): the plug-in pair, the posterior
    // estimand, then its uncertainty columns, in the order the docs list them.
    "linear_predictor_plugin",
    "mean_plugin",
    "posterior_mean",
    "linear_predictor_standard_error",
    "posterior_mean_standard_error",
    "posterior_mean_lower",
    "posterior_mean_upper",
    // Class-specific schema retained by the transformation-normal and
    // Bernoulli marginal-slope classes.
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

/// Select a skip-transcoder style integer rank in `[0, max_rank]` with the
/// continuous log-hyperparameters profiled at every visited rank (#3920).
///
/// `profile(evaluation_id, rank, log_hyperparameters, transition, warm_start)`
/// is the inner fit: it fits rank `rank` at `log_hyperparameters` to
/// convergence and returns `(value, gradient, gradient_scale)`, the negative
/// log evidence, its analytic gradient in the log-hyperparameters and the
/// positive magnitude of the terms that gradient is summed from.
/// `transition` is `"seed"`, `"birth"`, `"death"` or `"continuous"`, and
/// `warm_start` is the `evaluation_id` of an earlier fit to start from (or
/// `None`). The BFGS profile, the stationarity certificate and the
/// birth/death walk all run in
/// `gam_solve::profiled_rank_selection`; an exception raised by `profile` is
/// re-raised unchanged.
///
/// Returns JSON: `{"status": "selected", "selected": E, "death": E | null,
/// "birth": E | null, "moves": [{"from_rank", "to_rank", "value_gap",
/// "accepted"}], "evaluations": n}`, or `{"status": "non_converged",
/// "evaluation": E, "reason": str}` when a rank's profile stopped without the
/// certificate. Each `E` is `{"evaluation_id", "rank", "log_hyperparameters",
/// "value", "gradient", "gradient_scale", "stationarity_defect",
/// "stationarity_tolerance"}`.
#[pyfunction]
fn select_rank_with_profiled_hyperparameters(
    initial_rank: usize,
    max_rank: usize,
    initial_log_hyperparameters: Vec<f64>,
    profile: &Bound<'_, PyAny>,
) -> PyResult<String> {
    use gam::solver::profiled_rank_selection::{
        ProfileEvaluation, ProfileQuery, ProfileSample, RankSelectionError,
        select_rank_with_profiled_hyperparameters as select,
    };

    fn evaluation_json(evaluation: &ProfileEvaluation) -> serde_json::Value {
        serde_json::json!({
            "evaluation_id": evaluation.evaluation_id,
            "rank": evaluation.rank,
            "log_hyperparameters": evaluation.log_hyperparameters.to_vec(),
            "value": evaluation.value,
            "gradient": evaluation.gradient.to_vec(),
            "gradient_scale": evaluation.gradient_scale,
            "stationarity_defect": evaluation.stationarity_defect(),
            "stationarity_tolerance": evaluation.stationarity_tolerance(),
        })
    }

    let outcome = select(
        initial_rank,
        max_rank,
        Array1::from_vec(initial_log_hyperparameters),
        |query: &ProfileQuery| -> PyResult<ProfileSample> {
            let (value, gradient, gradient_scale) = profile
                .call1((
                    query.evaluation_id,
                    query.rank,
                    query.log_hyperparameters.to_vec(),
                    query.transition.as_str(),
                    query.warm_start,
                ))?
                .extract::<(f64, Vec<f64>, f64)>()?;
            Ok(ProfileSample {
                value,
                gradient: Array1::from_vec(gradient),
                gradient_scale,
            })
        },
    );
    let out = match outcome {
        Ok(selection) => serde_json::json!({
            "status": "selected",
            "selected": evaluation_json(&selection.selected),
            "death": selection.death.as_ref().map(evaluation_json),
            "birth": selection.birth.as_ref().map(evaluation_json),
            "moves": selection
                .moves
                .iter()
                .map(|step| {
                    serde_json::json!({
                        "from_rank": step.from_rank,
                        "to_rank": step.to_rank,
                        "value_gap": step.value_gap,
                        "accepted": step.accepted,
                    })
                })
                .collect::<Vec<_>>(),
            "evaluations": selection.evaluations,
        }),
        Err(RankSelectionError::Oracle(error)) => return Err(error),
        Err(RankSelectionError::NonConvergence { evaluation, reason }) => serde_json::json!({
            "status": "non_converged",
            "evaluation": evaluation_json(&evaluation),
            "reason": reason,
        }),
        Err(error @ (RankSelectionError::InvalidRequest(_)
        | RankSelectionError::InvalidSample { .. })) => {
            return Err(py_value_error(format!(
                "select_rank_with_profiled_hyperparameters: {error}"
            )));
        }
    };
    serde_json::to_string(&out).map_err(|err| {
        py_value_error(format!(
            "select_rank_with_profiled_hyperparameters: serialise: {err}"
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

/// Stacked response-scale predictive mean `Σ_k w_k μ_k(x)` over the candidates a
/// `TopologyStack` predicts with; `means` is indexed `[candidate][row]`.
#[pyfunction]
fn stacked_predictive_mean(weights: Vec<f64>, means: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    gam::solver::topology_stack_gaussian::stacked_predictive_mean(&weights, &means)
        .map_err(py_value_error)
}

// Each lookup below names the one `SummaryPayload` field that publishes the
// quantity; there are no alternative spellings to probe.
const REML_SCORE_KEYS: &[&str] = &["reml_score"];

const RAW_REML_SCORE_KEYS: &[&str] = &["raw_reml_score"];

/// Payload key carrying WHY a summary has no comparable criterion (#2595,
/// #2627). Present exactly when `reml_score` is `null`.
const REML_UNAVAILABLE_KEYS: &[&str] = &["reml_score_unavailable"];

/// The refusal a ranking surface raises when the summary it was handed has no
/// criterion to rank.
///
/// Reads the payload's own recorded reason when there is one, so the user is
/// told what actually happened to their fit rather than that a field is
/// "missing" — the field is present, and it is `null` on purpose.
fn no_criterion_error(payload: &serde_json::Value, surface: &str) -> pyo3::PyErr {
    match json_lookup_str(payload, REML_UNAVAILABLE_KEYS) {
        Some(reason) => py_value_error(format!("{surface}: {reason}")),
        None => py_value_error(format!(
            "{surface}: this model summary carries no reml_score field"
        )),
    }
}

enum RemlFitView<'py> {
    /// The `SummaryPayload` of a gamfit Model or its saved bytes.
    SavedSummary(serde_json::Value),
    /// A summary mapping (a dict or `gamfit.results.Summary`) read through `.get`.
    Mapping(Bound<'py, PyAny>),
}

#[pyfunction]
fn extract_reml_score_raw(py: Python<'_>, fit: Py<PyAny>) -> PyResult<f64> {
    let fit = fit.bind(py);
    extract_reml_score_raw_impl(fit)
}

/// Rank fitted models on their smoothing-corrected AIC. The ranking is
/// `compare_saved_models`, the same one `gam compare` prints.
#[pyfunction(signature = (fits, names = None))]
fn compare_models(
    py: Python<'_>,
    fits: Vec<Py<PyAny>>,
    names: Option<Vec<String>>,
) -> PyResult<PyObject> {
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
    let model_bytes = fits
        .iter()
        .map(|fit| {
            let fit = fit.bind(py);
            if let Ok(bytes) = fit.extract::<Vec<u8>>() {
                return Ok(bytes);
            }
            if fit.hasattr("_model_bytes")? {
                return fit.getattr("_model_bytes")?.extract::<Vec<u8>>();
            }
            Err(PyTypeError::new_err(format!(
                "compare_models: expected a gamfit.Model or its saved bytes; got {}",
                fit.get_type().name()?
            )))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let models = detach_typed_py_result(
        py,
        "compare_models",
        move || {
            model_bytes
                .iter()
                .map(|bytes| load_model_impl(bytes))
                .collect::<Result<Vec<_>, _>>()
        },
        saved_model_error_to_pyerr,
    )?;
    let comparison = detach_py_result(py, "compare_models", move || {
        let named = labels
            .into_iter()
            .zip(models.iter())
            .collect::<Vec<_>>();
        let comparison = compare_saved_models(&named)?;
        serde_json::to_value(comparison)
            .map_err(|err| format!("failed to serialize model comparison: {err}"))
    })?;
    json_value_to_py(py, &comparison)
}

fn extract_reml_score_raw_impl(fit: &Bound<'_, PyAny>) -> PyResult<f64> {
    let view = reml_fit_view(fit)?;
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
        RemlFitView::SavedSummary(payload) => Err(no_criterion_error(payload, "compare_models")),
        RemlFitView::Mapping(fit) => Err(PyTypeError::new_err(format!(
            "compare_models: cannot extract reml_score from {}; pass a gamfit.Model \
             or a summary mapping with 'reml_score'",
            fit.get_type().name()?
        ))),
    }
}

fn extract_float_metadata_from_view(
    view: &RemlFitView<'_>,
    keys: &[&str],
) -> PyResult<Option<f64>> {
    match view {
        RemlFitView::SavedSummary(payload) => Ok(json_lookup_f64(payload, keys)),
        RemlFitView::Mapping(_) => {
            let Some(value) = extract_py_metadata_value(view, keys)? else {
                return Ok(None);
            };
            value.extract::<f64>().map(Some)
        }
    }
}

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

fn reml_fit_view<'py>(fit: &Bound<'py, PyAny>) -> PyResult<RemlFitView<'py>> {
    if let Ok(model_bytes) = fit.extract::<Vec<u8>>() {
        let model =
            load_model_impl(&model_bytes).map_err(|err| saved_model_error_to_pyerr(fit.py(), err))?;
        let summary = summary_payload_value(&model).map_err(PyValueError::new_err)?;
        return Ok(RemlFitView::SavedSummary(summary));
    }
    if fit.hasattr("_prediction_model")? {
        let compiled = fit.getattr("_prediction_model")?;
        let compiled = compiled.cast::<PyFittedModel>()?;
        return Ok(RemlFitView::SavedSummary(
            compiled.get().summary_value()?.clone(),
        ));
    }
    if fit.hasattr("get")? && fit.getattr("get")?.is_callable() {
        return Ok(RemlFitView::Mapping(fit.clone()));
    }
    Err(PyTypeError::new_err(format!(
        "compare_models: expected a gamfit.Model, its saved bytes, or a summary mapping; got {}",
        fit.get_type().name()?
    )))
}

fn extract_py_metadata_value<'py>(
    view: &RemlFitView<'py>,
    keys: &[&str],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let RemlFitView::Mapping(mapping) = view else {
        return Ok(None);
    };
    for key in keys {
        let value = mapping.call_method1("get", (*key,))?;
        if !value.is_none() {
            return Ok(Some(value));
        }
    }
    Ok(None)
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

fn json_number_to_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|value| value as f64))
        .or_else(|| value.as_u64().map(|value| value as f64))
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
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let fit = detach_pyresult(py, "gaussian_reml_fit", move || {
        let gated_x = gate_design_for_forward(
            x_values.view(),
            by_values.as_ref().map(|b| b.view()),
            by_start_col,
        )
        .map_err(py_value_error)?;
        let fit_x = gated_x.as_ref().map_or(x_values.view(), |g| g.view());
        let gated_weights = gate_weights_for_forward(
            weight_values.as_ref().map(|w| w.view()),
            by_values.as_ref().map(|b| b.view()),
            x_values.nrows(),
        )
        .map_err(py_value_error)?;
        // A singular XᵀWX (p > n, or rank-deficient) is fit through the penalty
        // pencil when the penalty identifies null(W½X) (gam#3366) and refused
        // with the engine's typed error otherwise (gam#3310).
        gaussian_reml_multi_closed_form_with_cache(
            fit_x,
            y_values.view(),
            penalty_values.view(),
            gated_weights.as_ref().map(|w| w.view()),
            init_lambda,
            None,
        )
        .map_err(estimation_error_to_pyerr)
    })?;
    let out = PyDict::new(py);
    set_ok_gaussian_reml_items(py, &out, fit)?;
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
/// Programmatic (formula-API-bypass) entry into the exact profiled Gaussian
/// REML criterion differentiated by `gaussian_reml_fit_blocks_backward`.
/// Each penalty block receives one λ_k and the joint coefficient map must be
/// identified by the weighted design plus the canonical penalty roots.
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
    // Running width: the prefix-sum is carried, never re-read off the tail.
    let mut p_total = 0usize;
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
        p_total += view.ncols();
        col_offsets.push(p_total);
    }
    if n_rows == 0 || p_total == 0 {
        return Err(py_value_error(
            "gaussian_reml_fit_blocks_forward requires non-empty rows and at least one coefficient column"
                .to_string(),
        ));
    }
    let designs_owned: Vec<Array2<f64>> = designs
        .iter()
        .map(|design| design.as_array().to_owned())
        .collect();
    let mut penalties_owned: Vec<Array2<f64>> = Vec::with_capacity(designs.len());
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
        penalties_owned.push(pv.to_owned());
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

    let weights_for_fit = weights_owned.clone();
    let fit = detach_estimation_result(py, "gaussian_reml_fit_blocks_forward", move || {
        gam::solver::gaussian_reml::gaussian_reml_fit_blocks_exact(
            &designs_owned,
            &penalties_owned,
            y_col.view(),
            Some(weights_for_fit.view()),
            heuristic_owned.as_deref(),
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

    let out = PyDict::new(py);
    out.set_item("coefficients", fit.coefficients.into_pyarray(py))?;
    out.set_item("fitted", fit.fitted.into_pyarray(py))?;
    out.set_item("lambdas", lambdas.into_pyarray(py))?;
    out.set_item("log_lambdas", fit.log_lambdas.into_pyarray(py))?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("edf", fit.edf.into_pyarray(py))?;
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
/// optional linear inequality system `A·β ≥ b`.
///
/// Wraps the constrained Gaussian REML driver
/// (`constrained_gaussian_reml_forward` over a `LinearInequalityConstraints`
/// system) that backs the formula-API shape constraints. Forward-only: no analytic VJP through the active-set
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
    let init_lambda = log_lambda_at_optimum
        .map(gam::checked_exp_log_strength)
        .transpose()
        .map_err(|error| {
            py_value_error(format!(
                "gaussian_reml_fit_with_constraints_backward: {error}"
            ))
        })?;

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
        "cache_data_null_basis",
        result.cache_data_null_basis.into_pyarray(py),
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

/// Read a position fit's `knots_or_centers`: `None`, an integer basis size, or
/// an explicit float64 vector. [`resolve_position_basis`] owns all three.
fn position_basis_locations_arg(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<PositionBasisLocations> {
    let Some(value) = value else {
        return Ok(PositionBasisLocations::Default);
    };
    if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
        let count: i64 = value.extract()?;
        let count = usize::try_from(count).map_err(|_| {
            py_value_error(format!(
                "knots_or_centers: an integer basis size must be non-negative, got {count}"
            ))
        })?;
        return Ok(PositionBasisLocations::Count(count));
    }
    if let Ok(given) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(PositionBasisLocations::Given(given.as_array().to_owned()));
    }
    let given: Vec<f64> = value.extract().map_err(|_| {
        PyTypeError::new_err(
            "knots_or_centers must be None, an integer basis size, or a 1-D float vector",
        )
    })?;
    Ok(PositionBasisLocations::Given(Array1::from_vec(given)))
}

/// Read a position fit's `penalty`: `None`, the name of the kind's canonical
/// penalty, or an explicit float64 matrix.
fn position_penalty_arg(value: Option<&Bound<'_, PyAny>>) -> PyResult<PositionPenaltyRequest> {
    let Some(value) = value else {
        return Ok(PositionPenaltyRequest::Canonical);
    };
    if value.is_instance_of::<PyString>() {
        return Ok(PositionPenaltyRequest::Named(value.extract()?));
    }
    let given: PyReadonlyArray2<'_, f64> = value.extract()?;
    Ok(PositionPenaltyRequest::Given(given.as_array().to_owned()))
}

/// The basis state a forward position fit ran on, returned so a caller can
/// replay the same basis at predict time.
fn set_position_basis_items(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    basis: ResolvedPositionBasis,
    periodic: bool,
) -> PyResult<()> {
    out.set_item("knots_or_centers", basis.locations.into_pyarray(py))?;
    out.set_item("penalty", basis.penalty.into_pyarray(py))?;
    out.set_item("basis_kind", basis.display_kind)?;
    out.set_item("basis_order", basis.order)?;
    out.set_item("periodic", periodic)?;
    out.set_item("period", basis.period)?;
    Ok(())
}

#[pyfunction(signature = (
    t,
    y,
    basis_kind = None,
    knots_or_centers = None,
    penalty = None,
    basis_order = None,
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
    basis_kind: Option<String>,
    knots_or_centers: Option<&Bound<'py, PyAny>>,
    penalty: Option<&Bound<'py, PyAny>>,
    basis_order: Option<usize>,
    periodic: bool,
    period: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let locations = position_basis_locations_arg(knots_or_centers)?;
    let penalty_request = position_penalty_arg(penalty)?;
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let (fit, basis) = detach_pyresult(py, "gaussian_reml_fit_positions", move || {
        let basis = resolve_position_basis(
            t_values.view(),
            basis_kind.as_deref(),
            locations,
            penalty_request,
            basis_order,
            periodic,
            period,
        )
        .map_err(py_value_error)?;
        let x = position_basis_design(
            t_values.view(),
            basis.locations.view(),
            basis.kind.engine_name(),
            basis.order,
            periodic,
            basis.period,
        )
        .map_err(py_value_error)?;
        let gated_x =
            gate_design_for_forward(x.view(), by_values.as_ref().map(|b| b.view()), by_start_col)
                .map_err(py_value_error)?;
        let fit_x = gated_x.as_ref().map_or(x.view(), |g| g.view());
        let gated_weights = gate_weights_for_forward(
            weight_values.as_ref().map(|w| w.view()),
            by_values.as_ref().map(|b| b.view()),
            x.nrows(),
        )
        .map_err(py_value_error)?;
        // A singular XᵀWX is fit through the penalty pencil when the penalty
        // identifies null(W½X) (gam#3366) and refused otherwise (gam#3310).
        let fit = gaussian_reml_multi_closed_form_with_cache(
            fit_x,
            y_values.view(),
            basis.penalty.view(),
            gated_weights.as_ref().map(|w| w.view()),
            init_lambda,
            None,
        )
        .map_err(estimation_error_to_pyerr)?;
        Ok((fit, basis))
    })?;
    let out = PyDict::new(py);
    set_ok_gaussian_reml_items(py, &out, fit)?;
    set_position_basis_items(py, &out, basis, periodic)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (
    t,
    y,
    basis_kind = None,
    knots_or_centers = None,
    penalty = None,
    grad_lambda = 0.0,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = 0.0,
    grad_edf = 0.0,
    forward_state = None,
    basis_order = None,
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
    basis_kind: Option<String>,
    knots_or_centers: Option<&Bound<'py, PyAny>>,
    penalty: Option<&Bound<'py, PyAny>>,
    grad_lambda: f64,
    grad_coefficients: Option<PyReadonlyArray2<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: f64,
    grad_edf: f64,
    forward_state: Option<&Bound<'py, PyDict>>,
    basis_order: Option<usize>,
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
    let locations = position_basis_locations_arg(knots_or_centers)?;
    let penalty_request = position_penalty_arg(penalty)?;
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let grad_coefficients_values = grad_coefficients.as_ref().map(|g| g.as_array().to_owned());
    let grad_fitted_values = grad_fitted.as_ref().map(|g| g.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let backward = detach_py_result(py, "gaussian_reml_fit_positions_backward", move || {
        let basis = resolve_position_basis(
            t_values.view(),
            basis_kind.as_deref(),
            locations,
            penalty_request,
            basis_order,
            periodic,
            period,
        )?;
        gaussian_reml_fit_positions_backward_impl(
            t_values.view(),
            y_values.view(),
            basis.locations.view(),
            basis.kind.engine_name(),
            basis.order,
            periodic,
            basis.period,
            basis.penalty.view(),
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
    basis_kind = None,
    knots_or_centers = None,
    penalty = None,
    basis_order = None,
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
    basis_kind: Option<String>,
    knots_or_centers: Option<&Bound<'py, PyAny>>,
    penalty: Option<&Bound<'py, PyAny>>,
    basis_order: Option<usize>,
    periodic: bool,
    period: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let locations = position_basis_locations_arg(knots_or_centers)?;
    let penalty_request = position_penalty_arg(penalty)?;
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let row_offset_values = row_offsets.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let by_values = by.as_ref().map(|b| b.as_array().to_owned());
    let (result, basis) = detach_py_result(py, "gaussian_reml_fit_positions_batched", move || {
        // The basis locations are placed on the concatenated positions of every group.
        let basis = resolve_position_basis(
            t_values.view(),
            basis_kind.as_deref(),
            locations,
            penalty_request,
            basis_order,
            periodic,
            period,
        )?;
        let result = gaussian_reml_fit_positions_batched_impl(
            t_values.view(),
            y_values.view(),
            row_offset_values.view(),
            basis.locations.view(),
            basis.kind.engine_name(),
            basis.order,
            periodic,
            basis.period,
            basis.penalty.view(),
            weight_values.as_ref().map(|w| w.view()),
            init_lambda,
            by_values.as_ref().map(|b| b.view()),
            by_start_col,
        )?;
        Ok((result, basis))
    })?;
    let out = PyDict::new(py);
    set_batched_gaussian_reml_dict_items(py, &out, result)?;
    set_position_basis_items(py, &out, basis, periodic)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (
    t,
    y,
    row_offsets,
    basis_kind = None,
    knots_or_centers = None,
    penalty = None,
    grad_lambda = None,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = None,
    grad_edf = None,
    forward_state = None,
    basis_order = None,
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
    basis_kind: Option<String>,
    knots_or_centers: Option<&Bound<'py, PyAny>>,
    penalty: Option<&Bound<'py, PyAny>>,
    grad_lambda: Option<PyReadonlyArray1<'py, f64>>,
    grad_coefficients: Option<PyReadonlyArray3<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: Option<PyReadonlyArray1<'py, f64>>,
    grad_edf: Option<PyReadonlyArray1<'py, f64>>,
    forward_state: Option<&Bound<'py, PyDict>>,
    basis_order: Option<usize>,
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
    let locations = position_basis_locations_arg(knots_or_centers)?;
    let penalty_request = position_penalty_arg(penalty)?;
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let row_offset_values = row_offsets.as_array().to_owned();
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
            let basis = resolve_position_basis(
                t_values.view(),
                basis_kind.as_deref(),
                locations,
                penalty_request,
                basis_order,
                periodic,
                period,
            )?;
            gaussian_reml_fit_positions_batched_backward_impl(
                t_values.view(),
                y_values.view(),
                row_offset_values.view(),
                basis.locations.view(),
                basis.kind.engine_name(),
                basis.order,
                periodic,
                basis.period,
                basis.penalty.view(),
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

#[cfg(test)]
mod prediction_payload_tests {
    use super::{
        PredictionPayload, SurvivalPredictionJsonPayload, SurvivalPredictionPayload,
        parse_covariance_mode,
    };
    use gam_predict::{InferenceCovarianceMode, PredictUncertaintyOptions};
    use std::collections::BTreeMap;

    #[test]
    fn public_covariance_modes_are_exact_and_default_to_required_smoothing() {
        assert_eq!(
            PredictUncertaintyOptions::default().covariance_mode,
            InferenceCovarianceMode::SmoothingCorrected
        );
        assert_eq!(parse_covariance_mode(None).expect("default mode"), None);
        assert_eq!(
            parse_covariance_mode(Some("conditional")).expect("conditional mode"),
            Some(InferenceCovarianceMode::Conditional)
        );
        assert_eq!(
            parse_covariance_mode(Some("smoothing")).expect("smoothing mode"),
            Some(InferenceCovarianceMode::SmoothingCorrected)
        );
        assert!(
            parse_covariance_mode(Some("required")).is_err(),
            "the removed compatibility spelling must not remain as a dead mode"
        );
    }

    #[test]
    fn model_based_prediction_payload_exposes_exact_covariance_source() {
        let payload = PredictionPayload {
            columns: BTreeMap::from([("mean".to_string(), vec![1.0])]),
            model_class: "standard".to_string(),
            point_column: "posterior_mean",
            point_shape: "estimand_explicit",
            point_columns: None,
            family: "identity".to_string(),
            interval_method: None,
            covariance_source: Some("smoothing-corrected".to_string()),
            point_covariance_source: Some("conditional".to_string()),
            point_covariance_note: None,
        };

        let value = serde_json::to_value(payload).expect("serialize prediction payload");
        assert_eq!(
            value
                .get("covariance_source")
                .and_then(|item| item.as_str()),
            Some("smoothing-corrected")
        );
        assert!(
            value.get("point_covariance_note").is_none(),
            "a point on the fit's own covariance carries no note"
        );
    }

    /// gam#2985: a withheld fit's posterior-mean point reaches Python with the
    /// typed provenance note beside its covariance source.
    #[test]
    fn a_withheld_fit_prediction_payload_carries_its_point_note_2985() {
        let declined = gam::estimate::CovarianceDeclined::
            BmsGeneratedRegressorResidualRepairChannelUnavailable {
                unavailable_channel: "the pin's missing channel".to_string(),
            };
        let note =
            gam_predict::PointCovarianceProvenance::ConditionalOnFittedLatentLaw { declined }.explain();
        let payload = PredictionPayload {
            columns: BTreeMap::from([("posterior_mean".to_string(), vec![0.4])]),
            model_class: "bernoulli marginal-slope".to_string(),
            point_column: "posterior_mean",
            point_shape: "estimand_explicit",
            point_columns: None,
            family: "probit".to_string(),
            interval_method: None,
            covariance_source: None,
            point_covariance_source: Some("conditional".to_string()),
            point_covariance_note: Some(note.clone()),
        };
        let value = serde_json::to_value(payload).expect("serialize prediction payload");
        assert_eq!(
            value.get("point_covariance_note").and_then(|item| item.as_str()),
            Some(note.as_str())
        );
        assert!(
            note.contains("conditional on the fitted latent law")
                && note.contains("the pin's missing channel"),
            "{note}"
        );
    }

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
            covariance_source: Some("smoothing-corrected".to_string()),
            rmst_tau: Some(2.0),
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
                .expect("the survival predict output declares a survival_prob column"),
            &vec![0.0, 0.0]
        );
    }
}
