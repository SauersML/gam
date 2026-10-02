//! Response-geometry GAMs (#2114), with the whole workflow in one Rust owner (#2899 P1).
//!
//! Manifold-valued responses (sphere, simplex, curved matrix and constant-curvature
//! geometries) are mapped to the tangent space at an intrinsic base point, the tangent
//! coordinates are fitted jointly as one vector-valued Gaussian GAM with one smoothing
//! parameter per smooth shared across every coordinate
//! ([`gam_models::response_geometry::fit_shared_tangent_reml`]), and predictions are
//! mapped back onto the manifold. For a constant-curvature geometry the curvature is an
//! estimand: `κ̂` is fitted from the responses first and the tangent chart is built at it.
//!
//! The fitted model is one container ([`ResponseGeometryModel`]) that every front door
//! saves, loads, predicts and summarizes through; it embeds the scalar template model
//! whose design the joint fit's coefficients act on.

use gam_data::{ColumnKindTag, EncodedDataset, SchemaColumn};
use gam_geometry::response_geometry::{ResponseManifold, exp_map, fit_response_curvature, log_map};
use gam_models::fit_orchestration::{FitConfig, FitRequest, WorkflowError, materialize};
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::response_geometry::{
    SharedTangentPenalty, SharedTangentRemlRequest, fit_shared_tangent_reml,
};
use gam_solve::estimate::EstimationError;
use ndarray::{Array1, Array2, ArrayView3};
use serde::{Deserialize, Serialize};

/// Schema tag of the saved response-geometry container.
pub const RESPONSE_GEOMETRY_SCHEMA: &str = "gamfit.ResponseGeometryModel/v1";

/// The column the scalar template model is fitted on: the first tangent coordinate.
/// Only its design is read; the coefficients come from the joint fit.
const TEMPLATE_RESPONSE_COLUMN: &str = "__gamfit_response_geometry_shared";

/// Coverage of the profile-likelihood interval reported for a fitted curvature.
const CURVATURE_INTERVAL_LEVEL: f64 = 0.95;

/// Why a response-geometry fit, load or prediction returned no result.
#[derive(Debug)]
pub enum ResponseGeometryError {
    /// The request, the data or a saved container is not usable.
    Invalid(String),
    /// The scalar template fit failed.
    Fit(WorkflowError),
    /// The joint tangent REML failed; its typed error carries resume evidence.
    Engine(EstimationError),
}

impl std::fmt::Display for ResponseGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Fit(error) => write!(f, "{error}"),
            Self::Engine(error) => write!(f, "{error}"),
        }
    }
}

impl From<String> for ResponseGeometryError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

/// What a response-geometry fit takes beyond the formula, the table and the scalar fit
/// configuration.
#[derive(Clone, Debug)]
pub struct ResponseGeometryRequest {
    /// Geometry label (`"sphere"`, `"simplex"`, `"alr"`, `"constant_curvature(dim=2)"`, …).
    pub geometry: String,
    /// The response component columns, in manifold-coordinate order.
    pub response_columns: Vec<String>,
    /// Simplex tangent chart (`"clr"`/`"alr"`); `None` takes the geometry's own.
    pub coordinates: Option<String>,
    /// ALR reference part.
    pub reference: isize,
    /// Optional Fisher-Rao precision coupling the tangent residuals: a `(N,)` scale, a
    /// shared `(D, D)` block or an `(N, D, D)` stack.
    pub fisher_rao_w: Option<ndarray::ArrayD<f64>>,
}

/// The curvature estimand of a constant-curvature fit (#944, #1104).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseCurvatureSummary {
    pub kappa_hat: f64,
    pub ci_level: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub ci_lo_at_bound: bool,
    pub ci_hi_at_bound: bool,
    pub verdict: String,
    pub flatness_lr: f64,
    pub flatness_pvalue: f64,
    pub railed_at_resolution_limit: bool,
    pub railed_at_hyperbolic_resolution_limit: bool,
    pub kappa_r2: f64,
    pub characteristic_radius: f64,
    pub base_point: Vec<f64>,
}

/// The joint tangent REML fit's reported quantities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedTangentFitReport {
    #[serde(default = "ok_status")]
    pub status: String,
    pub reml_score: f64,
    /// Shared per-smooth smoothing parameters.
    pub lambdas: Vec<f64>,
    /// Shared per-smooth effective degrees of freedom.
    pub edf: Vec<f64>,
    /// The pooled isotropic residual variance, as a one-element list.
    pub sigma2: Vec<f64>,
}

fn ok_status() -> String {
    "ok".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SharedTangentRecord {
    template_model_b64: String,
    /// `K × D`, row-major rows.
    coefficients: Vec<Vec<f64>>,
    fit: SharedTangentFitReport,
}

/// A fitted response-geometry GAM: the geometry its tangent chart was built on, the chart
/// origin, the scalar template whose design the joint coefficients act on, and the joint
/// fit.
#[derive(Clone)]
pub struct ResponseGeometryModel {
    /// The geometry the log/exp maps use; a constant-curvature fit carries its `κ̂`.
    pub response_geometry: String,
    pub response_columns: Vec<String>,
    pub base_point: Array1<f64>,
    /// The resolved tangent chart label.
    pub coordinates: String,
    pub reference: isize,
    /// The table kind the model was trained on, echoed to shape predictions.
    pub training_table_kind: String,
    pub curvature: Option<ResponseCurvatureSummary>,
    template: FittedModel,
    /// `K × D`.
    coefficients: Array2<f64>,
    pub fit: SharedTangentFitReport,
}

#[derive(Serialize, Deserialize)]
struct ResponseGeometryDocument {
    schema: String,
    response_geometry: String,
    response_columns: Vec<String>,
    base_point: Vec<f64>,
    coordinates: String,
    #[serde(default = "default_reference")]
    reference: isize,
    training_table_kind: String,
    #[serde(default)]
    curvature: Option<ResponseCurvatureSummary>,
    /// Per-coordinate scalar models of the retired unshared fit; always empty.
    #[serde(default)]
    coordinate_models_b64: Vec<String>,
    shared_tangent_fit: SharedTangentRecord,
}

fn default_reference() -> isize {
    -1
}

fn column(dataset: &EncodedDataset, name: &str) -> Result<Array1<f64>, String> {
    let index = dataset
        .headers
        .iter()
        .position(|header| header == name)
        .ok_or_else(|| format!("response geometry column missing from data: {name:?}"))?;
    Ok(dataset.values.column(index).to_owned())
}

/// `dataset` with one more continuous column.
fn with_column(dataset: &EncodedDataset, name: &str, values: &Array1<f64>) -> Result<EncodedDataset, String> {
    if dataset.headers.iter().any(|header| header == name) {
        return Err(format!("response geometry reserved column already exists: {name}"));
    }
    let (rows, cols) = dataset.values.dim();
    let mut augmented = Array2::<f64>::zeros((rows, cols + 1));
    augmented.slice_mut(ndarray::s![.., ..cols]).assign(&dataset.values);
    augmented.column_mut(cols).assign(values);
    let mut schema = dataset.schema.clone();
    schema.columns.push(SchemaColumn {
        name: name.to_string(),
        kind: ColumnKindTag::Continuous,
        levels: Vec::new(),
    });
    let mut headers = dataset.headers.clone();
    headers.push(name.to_string());
    let mut column_kinds = dataset.column_kinds.clone();
    column_kinds.push(ColumnKindTag::Continuous);
    Ok(EncodedDataset {
        headers,
        values: augmented,
        schema,
        column_kinds,
    })
}

/// The right-hand side of `response ~ terms`.
fn formula_rhs(formula: &str) -> Result<&str, String> {
    match formula.split_once('~') {
        Some((_, rhs)) if !rhs.trim().is_empty() => Ok(rhs.trim()),
        _ => Err("response-geometry formula must have the form 'response ~ terms'".to_string()),
    }
}

fn gaussian_identity(fit_config: &FitConfig) -> FitConfig {
    let mut config = fit_config.clone();
    config.family = Some("gaussian".to_string());
    config.link = Some("identity".to_string());
    config
}

/// A joint tangent REML fit on a formula's design.
#[derive(Clone, Debug)]
pub struct SharedTangentFormulaFit {
    pub report: SharedTangentFitReport,
    /// `K × D`.
    pub coefficients: Array2<f64>,
    /// `N × D` fitted tangent values.
    pub fitted: Array2<f64>,
}

/// The joint tangent REML on a formula's design: one smoothing parameter per smooth,
/// shared across every tangent coordinate, and an optional Fisher-Rao metric coupling
/// the coordinate residuals. A fit whose criterion, coefficients or smoothing
/// parameters are not finite reports `status = "diverged"`.
pub fn fit_shared_tangent_formula(
    dataset: &EncodedDataset,
    formula: &str,
    tangent: &Array2<f64>,
    fit_config: &FitConfig,
    fisher_rao_w: Option<ArrayView3<'_, f64>>,
) -> Result<SharedTangentFormulaFit, ResponseGeometryError> {
    let config = gaussian_identity(fit_config);
    let materialized = materialize(formula, dataset, &config).map_err(|err| err.to_string())?;
    let FitRequest::Standard(standard) = materialized.request else {
        return Err(ResponseGeometryError::Invalid(
            "shared-tangent Gaussian REML fitting requires a standard Gaussian formula".to_string(),
        ));
    };
    if !standard.family.is_gaussian_identity() {
        return Err(ResponseGeometryError::Invalid(
            "shared-tangent Gaussian REML fitting requires Gaussian identity".to_string(),
        ));
    }
    if standard.wiggle.is_some() {
        return Err(ResponseGeometryError::Invalid(
            "shared-tangent Gaussian REML fitting does not support link wiggle".to_string(),
        ));
    }
    if standard.offset.iter().any(|value| value.abs() > 0.0) {
        return Err(ResponseGeometryError::Invalid(
            "response geometry shared REML does not support offsets".to_string(),
        ));
    }
    // The engine consumes the design through bounded row chunks and streams the exact
    // joint sufficient statistics, so neither the stacked `(N·D) × (K·D)` system nor the
    // `Sᵇ ⊗ I_D` penalties are materialized here.
    let design = gam_terms::smooth::build_term_collection_design(standard.data.view(), &standard.spec)
        .map_err(|err| format!("failed to build formula design matrix: {err}"))?;
    if design.affine_offset.iter().any(|value| *value != 0.0) {
        return Err(ResponseGeometryError::Invalid(
            "shared-tangent Gaussian REML fitting does not support non-zero smooth anchors: the \
             vector-valued tangent response requires an explicit affine offset per tangent \
             coordinate"
                .to_string(),
        ));
    }
    let penalties = design
        .penalties
        .iter()
        .map(|penalty| SharedTangentPenalty::new(penalty.col_range.start, penalty.local.clone()))
        .collect();
    let request = SharedTangentRemlRequest::new(
        design.design,
        tangent.clone(),
        (*standard.weights).clone(),
        fisher_rao_w.map(|metric| metric.to_owned()),
        penalties,
    );
    let fit = fit_shared_tangent_reml(request).map_err(ResponseGeometryError::Engine)?;
    let finite = fit.reml_score.is_finite()
        && fit.coefficients.iter().all(|value| value.is_finite())
        && fit.lambdas.iter().all(|value| value.is_finite());
    Ok(SharedTangentFormulaFit {
        report: SharedTangentFitReport {
            status: if finite { ok_status() } else { "diverged".to_string() },
            reml_score: fit.reml_score,
            lambdas: fit.lambdas.to_vec(),
            edf: fit.edf_by_penalty.to_vec(),
            sigma2: vec![fit.sigma2],
        },
        coefficients: fit.coefficients,
        fitted: fit.fitted,
    })
}

/// The request-document fields a response-geometry fit cannot honour, with the name each
/// carries at the front doors. The joint tangent REML materializes the formula and the
/// weights alone, so any other model field would reach the template the predictions are
/// designed from but not the fit its coefficients come from; it is refused instead of
/// fitting a different model than the one requested.
const UNSUPPORTED_REQUEST_FIELDS: &[(&str, &str)] = &[
    ("link", "link"),
    ("flexible_link", "flexible_link"),
    ("offset", "offset"),
    ("noise_offset", "noise_offset"),
    ("noise_formula", "noise_formula"),
    ("slope_formula", "slope_formula"),
    ("z_column", "z_column"),
    ("residual_columns", "residual_columns"),
    ("survival_likelihood", "survival_likelihood"),
    ("survival_time_anchor", "survival_time_anchor"),
    ("baseline_target", "baseline_target"),
    ("baseline_scale", "baseline_scale"),
    ("baseline_shape", "baseline_shape"),
    ("baseline_rate", "baseline_rate"),
    ("baseline_makeham", "baseline_makeham"),
    ("frailty_kind", "frailty_kind"),
    ("frailty_sd", "frailty_sd"),
    ("hazard_loading", "hazard_loading"),
    ("transformation_normal", "transformation_normal"),
    ("ctn_stage1", "transformation_normal_stage1"),
    ("frozen_ctn", "transformation_normal_stage1"),
    ("expectile_tau", "expectile_tau"),
    ("scale_dimensions", "scale_dimensions"),
    ("firth", "firth"),
    ("latent_coordinates", "latents"),
    ("analytic_penalties", "penalties"),
    ("smooth_descriptors", "smooths"),
    ("precision_hyperpriors", "precision_hyperpriors"),
];

/// The front-door names of the unsupported fields `document` sets. A field is set when it
/// holds anything but `null` or `false`; `family` is set when it names anything but
/// `auto`, since the tangent coordinates are Gaussian identity by construction.
fn unsupported_request_fields(document: &serde_json::Map<String, serde_json::Value>) -> Vec<&'static str> {
    let is_set = |value: &serde_json::Value| {
        !matches!(value, serde_json::Value::Null | serde_json::Value::Bool(false))
    };
    let mut names: Vec<&'static str> = Vec::new();
    if document
        .get("family")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|family| !family.eq_ignore_ascii_case("auto"))
    {
        names.push("family");
    }
    for &(key, name) in UNSUPPORTED_REQUEST_FIELDS {
        if document.get(key).is_some_and(is_set) && !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Fit a response-geometry GAM.
///
/// `config_json` is the scalar fit request document (`gam.fit-request` config); the
/// tangent coordinates are fitted as a Gaussian identity GAM, so it carries the weights
/// and nothing that names another model (`UNSUPPORTED_REQUEST_FIELDS`).
pub fn fit_response_geometry(
    dataset: &EncodedDataset,
    formula: &str,
    request: &ResponseGeometryRequest,
    config_json: Option<&str>,
) -> Result<ResponseGeometryModel, ResponseGeometryError> {
    let document: serde_json::Map<String, serde_json::Value> = match config_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str(raw)
            .map_err(|err| format!("invalid fit config object: {err}"))?,
        _ => serde_json::Map::new(),
    };
    let unsupported = unsupported_request_fields(&document);
    if !unsupported.is_empty() {
        return Err(ResponseGeometryError::Invalid(format!(
            "{} {} not supported with response_geometry",
            unsupported.join(", "),
            if unsupported.len() == 1 { "is" } else { "are" }
        )));
    }
    let fit_config = gam_config::parse_fit_config_json(config_json)?;
    let fit_config = &fit_config;
    let n_columns = request.response_columns.len();
    if n_columns == 0 {
        return Err(ResponseGeometryError::Invalid(
            "response geometry needs at least one response column".to_string(),
        ));
    }
    let mut y = Array2::<f64>::zeros((dataset.values.nrows(), n_columns));
    for (index, name) in request.response_columns.iter().enumerate() {
        y.column_mut(index).assign(&column(dataset, name)?);
    }
    // Curvature as an estimand (#944, #1104): κ̂ comes from the responses, and the tangent
    // chart is built at it.
    let label = request.geometry.trim().to_ascii_lowercase();
    let (geometry, curvature) = if label.split('(').next() == Some("constant_curvature") {
        let dim = match ResponseManifold::parse(&label, n_columns)? {
            ResponseManifold::ConstantCurvature { dim, .. } => dim,
            other => {
                return Err(ResponseGeometryError::Invalid(format!(
                    "constant-curvature geometry parsed as {}",
                    other.canonical_label()
                )));
            }
        };
        let fit = fit_response_curvature(y.view(), dim, CURVATURE_INTERVAL_LEVEL)
            .map_err(|error| error.to_string())?;
        let verdict = match fit.profile_ci.verdict {
            gam_geometry::CurvatureVerdict::Spherical => "spherical",
            gam_geometry::CurvatureVerdict::Hyperbolic => "hyperbolic",
            gam_geometry::CurvatureVerdict::Flat => "flat",
        };
        let summary = ResponseCurvatureSummary {
            kappa_hat: fit.kappa_hat,
            ci_level: CURVATURE_INTERVAL_LEVEL,
            ci_lo: fit.profile_ci.ci_lo,
            ci_hi: fit.profile_ci.ci_hi,
            ci_lo_at_bound: fit.profile_ci.lo_at_bound,
            ci_hi_at_bound: fit.profile_ci.hi_at_bound,
            verdict: verdict.to_string(),
            flatness_lr: fit.flatness.lr_stat,
            flatness_pvalue: fit.flatness.p_value,
            railed_at_resolution_limit: fit.railed_at_resolution_limit,
            railed_at_hyperbolic_resolution_limit: fit.railed_at_hyperbolic_resolution_limit,
            kappa_r2: fit.kappa_r2,
            characteristic_radius: fit.characteristic_radius,
            base_point: fit.base.to_vec(),
        };
        (
            format!("constant_curvature(dim={n_columns},kappa={:?})", fit.kappa_hat),
            Some(summary),
        )
    } else {
        (label, None)
    };
    // The chart origin of a weighted fit is the weighted intrinsic mean, where the
    // weighted mass lives (#2125); the tangent regression reads the same weights.
    let weights = fit_config
        .weight_column
        .as_deref()
        .map(|name| column(dataset, name))
        .transpose()?;
    let (tangent, base_point, coordinates) = log_map(
        y.view(),
        &geometry,
        None,
        request.coordinates.as_deref(),
        request.reference,
        weights.as_ref().map(|w| w.view()),
    )?;
    let fisher = request
        .fisher_rao_w
        .as_ref()
        .map(|metric| {
            gam_problem::fisher_rao::normalize_fisher_rao_blocks(
                metric.view(),
                tangent.nrows(),
                tangent.ncols(),
            )
        })
        .transpose()?;
    let augmented = with_column(dataset, TEMPLATE_RESPONSE_COLUMN, &tangent.column(0).to_owned())?;
    let template_formula = format!("{TEMPLATE_RESPONSE_COLUMN} ~ {}", formula_rhs(formula)?);
    let template = FittedModel::from_payload(
        fit_formula_to_payload(template_formula.clone(), &augmented, &gaussian_identity(fit_config))
            .map_err(ResponseGeometryError::Fit)?,
    );
    let joint = fit_shared_tangent_formula(
        &augmented,
        &template_formula,
        &tangent,
        fit_config,
        fisher.as_ref().map(|metric| metric.view()),
    )?;
    if joint.report.status != ok_status() {
        return Err(ResponseGeometryError::Invalid(format!(
            "joint tangent Gaussian REML failed with status={:?}",
            joint.report.status
        )));
    }
    let (fit, coefficients) = (joint.report, joint.coefficients);
    Ok(ResponseGeometryModel {
        response_geometry: geometry,
        response_columns: request.response_columns.clone(),
        base_point,
        coordinates,
        reference: request.reference,
        training_table_kind: fit_config.training_table_kind.clone(),
        curvature,
        template,
        coefficients,
        fit,
    })
}

impl ResponseGeometryModel {
    /// The scalar template whose design the joint coefficients act on; a prediction
    /// table is projected onto its schema.
    pub fn template(&self) -> &FittedModel {
        &self.template
    }

    /// The tangent dimension `D`.
    pub fn tangent_dimension(&self) -> usize {
        self.coefficients.ncols()
    }

    /// `(response, tangent)` at the rows of `dataset`, which is in the template's schema.
    pub fn predict(&self, dataset: EncodedDataset) -> Result<(Array2<f64>, Array2<f64>), String> {
        let design = crate::partial_effect::standard_mean_design(&self.template, dataset)?;
        if design.ncols() != self.coefficients.nrows() {
            return Err(format!(
                "response-geometry design has {} columns but the joint fit has {} coefficient rows",
                design.ncols(),
                self.coefficients.nrows()
            ));
        }
        let tangent = design.dot(&self.coefficients);
        let response = exp_map(
            tangent.view(),
            &self.response_geometry,
            self.base_point.view(),
            Some(&self.coordinates),
            self.reference,
        )?;
        Ok((response, tangent))
    }

    /// The model summary: the geometry, the chart, the joint fit and its template's own
    /// summary, and the curvature estimand when there is one.
    pub fn summary(&self) -> Result<serde_json::Value, String> {
        let template = crate_summary(&self.template)?;
        Ok(serde_json::json!({
            "model_class": "response-geometry",
            "response_geometry": self.response_geometry,
            "response_columns": self.response_columns,
            "base_point": self.base_point.to_vec(),
            "coordinates": self.coordinates,
            "tangent_dimension": self.tangent_dimension(),
            "shared_smoothing": true,
            "coordinate_summaries": [],
            "shared_fit": {
                "model_class": "joint-tangent-gaussian-reml",
                "reml_score": self.fit.reml_score,
                "lambdas": self.fit.lambdas,
                "edf": self.fit.edf,
                "sigma2": self.fit.sigma2,
                "shared_smoothing": true,
                "template": template,
            },
            "curvature": self.curvature,
        }))
    }

    /// The saved container.
    pub fn to_saved_bytes(&self) -> Result<Vec<u8>, String> {
        let template = self.template.to_saved_bytes().map_err(|err| err.to_string())?;
        let document = ResponseGeometryDocument {
            schema: RESPONSE_GEOMETRY_SCHEMA.to_string(),
            response_geometry: self.response_geometry.clone(),
            response_columns: self.response_columns.clone(),
            base_point: self.base_point.to_vec(),
            coordinates: self.coordinates.clone(),
            reference: self.reference,
            training_table_kind: self.training_table_kind.clone(),
            curvature: self.curvature.clone(),
            coordinate_models_b64: Vec::new(),
            shared_tangent_fit: SharedTangentRecord {
                template_model_b64: base64_encode(&template),
                coefficients: self.coefficients.outer_iter().map(|row| row.to_vec()).collect(),
                fit: self.fit.clone(),
            },
        };
        serde_json::to_vec(&document).map_err(|err| format!("failed to serialize response geometry model: {err}"))
    }

    /// A saved container.
    pub fn from_saved_bytes(bytes: &[u8]) -> Result<Self, String> {
        let document: ResponseGeometryDocument = serde_json::from_slice(bytes)
            .map_err(|err| format!("invalid response-geometry model: {err}"))?;
        if document.schema != RESPONSE_GEOMETRY_SCHEMA {
            return Err(format!(
                "response-geometry model schema {:?} is not {RESPONSE_GEOMETRY_SCHEMA:?}",
                document.schema
            ));
        }
        if !document.coordinate_models_b64.is_empty() {
            return Err(
                "this response-geometry model carries per-coordinate models from the retired \
                 unshared fit; refit it"
                    .to_string(),
            );
        }
        let template_bytes = base64_decode(&document.shared_tangent_fit.template_model_b64)?;
        let template = FittedModel::from_saved_bytes(&template_bytes).map_err(|err| err.to_string())?;
        let rows = document.shared_tangent_fit.coefficients;
        let width = rows.first().map_or(0, Vec::len);
        if width == 0 || rows.iter().any(|row| row.len() != width) {
            return Err("response-geometry coefficients are not a non-empty K × D matrix".to_string());
        }
        let coefficients = Array2::from_shape_vec((rows.len(), width), rows.concat())
            .map_err(|err| err.to_string())?;
        Ok(Self {
            response_geometry: document.response_geometry,
            response_columns: document.response_columns,
            base_point: Array1::from(document.base_point),
            coordinates: document.coordinates,
            reference: document.reference,
            training_table_kind: document.training_table_kind,
            curvature: document.curvature,
            template,
            coefficients,
            fit: document.shared_tangent_fit.fit,
        })
    }
}

fn crate_summary(model: &FittedModel) -> Result<serde_json::Value, String> {
    let summary = gam_models::inference::saved_summary::saved_model_summary(model)
        .map_err(|err| err.to_string())?;
    serde_json::to_value(&summary).map_err(|err| format!("failed to serialize template summary: {err}"))
}

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard padded base64, the container's encoding of its embedded template.
fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let triple = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for position in 0..4 {
            if position <= chunk.len() {
                out.push(char::from(BASE64_ALPHABET[((triple >> (18 - 6 * position)) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    let text = text.trim();
    if text.len() % 4 != 0 {
        return Err("response-geometry template archive is not padded base64".to_string());
    }
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    for quad in text.as_bytes().chunks(4) {
        let mut triple = 0u32;
        let mut padding = 0usize;
        for (position, &byte) in quad.iter().enumerate() {
            let value = match byte {
                b'=' if position >= 2 => {
                    padding += 1;
                    0
                }
                _ if padding > 0 => {
                    return Err("response-geometry template archive has data after padding".to_string());
                }
                _ => BASE64_ALPHABET
                    .iter()
                    .position(|&symbol| symbol == byte)
                    .ok_or_else(|| {
                        "response-geometry template archive holds a non-base64 byte".to_string()
                    })? as u32,
            };
            triple = (triple << 6) | value;
        }
        let bytes = [(triple >> 16) as u8, (triple >> 8) as u8, triple as u8];
        out.extend_from_slice(&bytes[..3 - padding]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips_every_tail_length() {
        for length in 0..16 {
            let bytes: Vec<u8> = (0..length).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(base64_decode(&base64_encode(&bytes)).expect("decodes"), bytes);
        }
        // RFC 4648 test vectors.
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
    }

    /// Unit-sphere responses whose direction turns with `x`, plus a small tilt.
    fn sphere_dataset(n: usize) -> EncodedDataset {
        let headers = ["x", "y0", "y1", "y2"].map(str::to_string).to_vec();
        let records = (0..n)
            .map(|row| {
                let x = -1.0 + 2.0 * (row as f64 + 0.5) / n as f64;
                let angle = 0.8 * x + 0.03 * (5.1 * row as f64).sin();
                let tilt = 0.2 + 0.02 * (3.7 * row as f64).cos();
                let v = [angle.cos(), angle.sin(), tilt];
                let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                csv::StringRecord::from(vec![
                    format!("{x:.17e}"),
                    format!("{:.17e}", v[0] / norm),
                    format!("{:.17e}", v[1] / norm),
                    format!("{:.17e}", v[2] / norm),
                ])
            })
            .collect();
        gam_data::encode_recordswith_inferred_schema(headers, records).expect("encode")
    }

    fn sphere_request() -> ResponseGeometryRequest {
        ResponseGeometryRequest {
            geometry: "sphere".to_string(),
            response_columns: ["y0", "y1", "y2"].map(str::to_string).to_vec(),
            coordinates: None,
            reference: -1,
            fisher_rao_w: None,
        }
    }

    /// #2899 P1 — the whole response-geometry workflow in Rust: predictions lie on the
    /// sphere, reproduce the training responses closely, and survive the saved container.
    #[test]
    fn sphere_fit_predicts_on_the_manifold_and_round_trips() {
        let dataset = sphere_dataset(80);
        let model = fit_response_geometry(&dataset, "y0 ~ s(x)", &sphere_request(), None)
            .expect("the sphere response fit");
        assert_eq!(model.tangent_dimension(), 3);
        let (response, tangent) = model.predict(dataset.clone()).expect("predict");
        assert_eq!(tangent.dim(), (80, 3));
        let mut worst_norm = 0.0_f64;
        let mut worst_error = 0.0_f64;
        for (row, predicted) in response.outer_iter().enumerate() {
            worst_norm = worst_norm.max((predicted.dot(&predicted).sqrt() - 1.0).abs());
            for (column, value) in predicted.iter().enumerate() {
                worst_error = worst_error.max((value - dataset.values[[row, column + 1]]).abs());
            }
        }
        assert!(worst_norm <= 1.0e-12, "a prediction leaves the sphere by {worst_norm:.3e}");
        assert!(worst_error <= 0.1, "the smooth misses the responses by {worst_error:.3e}");

        let bytes = model.to_saved_bytes().expect("save");
        let loaded = ResponseGeometryModel::from_saved_bytes(&bytes).expect("load");
        let (reloaded, _) = loaded.predict(dataset).expect("predict after load");
        assert_eq!(reloaded, response);
        assert_eq!(loaded.response_columns, model.response_columns);
        let summary = loaded.summary().expect("summary");
        assert_eq!(summary["shared_smoothing"], serde_json::Value::Bool(true));
    }

    /// A request field the joint tangent fit cannot see is refused by its front-door name.
    #[test]
    fn a_request_field_the_joint_fit_cannot_honour_is_refused() {
        let dataset = sphere_dataset(20);
        for (document, name) in [
            (r#"{"offset": "x"}"#, "offset"),
            (r#"{"family": "poisson"}"#, "family"),
            (r#"{"smooth_descriptors": {"x": {}}}"#, "smooths"),
            (r#"{"firth": true}"#, "firth"),
        ] {
            let refusal =
                fit_response_geometry(&dataset, "y0 ~ s(x)", &sphere_request(), Some(document))
                    .err()
                    .map(|error| error.to_string());
            assert_eq!(
                refusal,
                Some(format!("{name} is not supported with response_geometry")),
                "{document}"
            );
        }
        // `family = "auto"` names no family, and a `false` flag sets nothing.
        assert!(
            fit_response_geometry(
                &dataset,
                "y0 ~ s(x)",
                &sphere_request(),
                Some(r#"{"family": "auto", "firth": false}"#),
            )
            .is_ok()
        );
    }

    #[test]
    fn formula_rhs_refuses_a_formula_without_terms() {
        assert_eq!(formula_rhs("y ~ s(x) + z").expect("rhs"), "s(x) + z");
        assert!(formula_rhs("y ~ ").is_err());
        assert!(formula_rhs("s(x)").is_err());
    }
}
