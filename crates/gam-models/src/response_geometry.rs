//! Exact shared-smoothing Gaussian REML for tangent-vector responses.
//!
//! The model has a scalar predictor design `X` (`N x K`) and a tangent
//! response `Y` (`N x D`).  A coefficient matrix `B` is fitted under
//!
//! `sum_i w_i (y_i - B' x_i)' M_i (y_i - B' x_i)
//!     + sum_b lambda_b tr(B' S_b B)`.
//!
//! The implementation deliberately never constructs the stacked
//! `(N D) x (K D)` design or a `S_b (x) I_D` penalty.  Isotropic metrics use
//! only `K x K` normal equations.  Varying Fisher metrics stream exact joint
//! sufficient statistics into a `(K D) x (K D)` Gram matrix whose storage is
//! independent of `N`.

use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_xt_diag_x, fast_xt_diag_y};
use gam_linalg::matrix::{DesignMatrix, LinearOperator};
use gam_linalg::utils::KahanSum;
// `DeclaredHessianForm`/`Derivative` originate in `gam_problem` and are only
// re-exported privately inside `gam_solve::rho_optimizer`; import them from the
// canonical source, matching every other `gam-models` outer-objective site.
use gam_problem::{DeclaredHessianForm, Derivative};
use gam_solve::estimate::EstimationError;
use gam_solve::rho_optimizer::{
    HessianValue, OuterCapability, OuterCriterionCertificate, OuterEval, OuterObjective,
    OuterProblem, SeedOutcome,
};
use ndarray::{Array1, Array2, Array3, s};
use serde::{Deserialize, Serialize};

use crate::inference::model::FittedModel;

const FIT_CONTEXT: &str = "shared-tangent Gaussian REML";
pub const RESPONSE_GEOMETRY_MODEL_VERSION: u32 = 1;

/// One compact predictor-space smoothing penalty.
///
/// `matrix` occupies the coefficient columns beginning at `column_start`.
/// The tangent-output identity factor is implicit and is never materialized.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedTangentPenalty {
    pub column_start: usize,
    pub matrix: Array2<f64>,
}

impl SharedTangentPenalty {
    pub fn new(column_start: usize, matrix: Array2<f64>) -> Self {
        Self {
            column_start,
            matrix,
        }
    }

    pub fn column_end(&self) -> usize {
        self.column_start + self.matrix.ncols()
    }
}

/// Owned request for a shared-tangent REML fit.
///
/// `design` may be dense, sparse, or operator-backed.  The fit consumes it
/// through bounded row chunks and therefore does not force materialization.
#[derive(Clone, Debug)]
pub struct SharedTangentRemlRequest {
    pub design: DesignMatrix,
    pub response: Array2<f64>,
    pub weights: Array1<f64>,
    pub fisher_metric: Option<Array3<f64>>,
    pub penalties: Vec<SharedTangentPenalty>,
    /// Optional log-lambda seed in the original `penalties` order.
    pub initial_log_lambdas: Option<Array1<f64>>,
}

impl SharedTangentRemlRequest {
    pub fn new(
        design: DesignMatrix,
        response: Array2<f64>,
        weights: Array1<f64>,
        fisher_metric: Option<Array3<f64>>,
        penalties: Vec<SharedTangentPenalty>,
    ) -> Self {
        Self {
            design,
            response,
            weights,
            fisher_metric,
            penalties,
            initial_log_lambdas: None,
        }
    }

    /// Convenience constructor for an owned dense design.
    pub fn from_dense(
        design: Array2<f64>,
        response: Array2<f64>,
        weights: Array1<f64>,
        fisher_metric: Option<Array3<f64>>,
        penalties: Vec<SharedTangentPenalty>,
    ) -> Self {
        Self::new(
            DesignMatrix::from(design),
            response,
            weights,
            fisher_metric,
            penalties,
        )
    }

    pub fn with_initial_log_lambdas(mut self, initial: Array1<f64>) -> Self {
        self.initial_log_lambdas = Some(initial);
        self
    }
}

/// A converged, serializable shared-tangent model.
///
/// This type is constructed only after the shared outer runner has produced a
/// successful analytic stationarity certificate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedTangentRemlFit {
    /// Predictor-by-output coefficient matrix (`K x D`).
    pub coefficients: Array2<f64>,
    /// Training fitted tangent vectors (`N x D`).
    pub fitted: Array2<f64>,
    /// Pooled residual dispersion `Q / (N·D - edf_total)`.
    pub sigma2: f64,
    /// Smoothing parameters in the request penalty order.  A numerically
    /// rank-zero penalty has no estimable smoothing coordinate and is `0`.
    pub lambdas: Array1<f64>,
    /// Per-penalty EDF in the request penalty order.
    pub edf_by_penalty: Array1<f64>,
    pub edf_total: f64,
    /// Minimized negative restricted log likelihood.
    pub reml_score: f64,
    pub n_observations: usize,
    pub n_outputs: usize,
    pub outer_iterations: usize,
    pub outer_certificate: OuterCriterionCertificate,
}

impl SharedTangentRemlFit {
    /// Predict tangent vectors from an operator-capable design.
    pub fn predict(&self, design: &DesignMatrix) -> Result<Array2<f64>, EstimationError> {
        predict_from_coefficients(design, &self.coefficients)
    }

    /// Convenience prediction entry point for an owned dense design.
    pub fn predict_dense(&self, design: Array2<f64>) -> Result<Array2<f64>, EstimationError> {
        self.predict(&DesignMatrix::from(design))
    }
}

/// Typed curvature-as-estimand record carried by a response-geometry archive.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResponseGeometryCurvature {
    pub kappa_hat: f64,
    pub confidence_level: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
    pub lower_at_bound: bool,
    pub upper_at_bound: bool,
    pub verdict: String,
    pub flatness_likelihood_ratio: f64,
    pub flatness_p_value: f64,
    pub railed_at_resolution_limit: bool,
    pub scale_free_kappa_radius_squared: f64,
    pub characteristic_radius: f64,
}

impl ResponseGeometryCurvature {
    fn validate(&self) -> Result<(), ResponseGeometryModelError> {
        let finite = [
            self.kappa_hat,
            self.confidence_level,
            self.confidence_lower,
            self.confidence_upper,
            self.flatness_likelihood_ratio,
            self.flatness_p_value,
            self.scale_free_kappa_radius_squared,
            self.characteristic_radius,
        ];
        if finite.iter().any(|value| !value.is_finite()) {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "curvature record contains non-finite values".to_string(),
            ));
        }
        if !(self.confidence_level > 0.0 && self.confidence_level < 1.0) {
            return Err(ResponseGeometryModelError::InvalidMetadata(format!(
                "curvature confidence level must lie in (0, 1), got {}",
                self.confidence_level
            )));
        }
        if self.confidence_lower > self.confidence_upper {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "curvature confidence interval is reversed".to_string(),
            ));
        }
        if !(self.flatness_likelihood_ratio >= 0.0
            && (0.0..=1.0).contains(&self.flatness_p_value)
            && self.characteristic_radius > 0.0)
        {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "curvature likelihood-ratio, p-value, or characteristic radius is invalid"
                    .to_string(),
            ));
        }
        if self.verdict.trim().is_empty() {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "curvature verdict must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Persistence and presentation metadata for [`ResponseGeometryModel`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResponseGeometryMetadata {
    pub response_geometry: String,
    pub response_columns: Vec<String>,
    pub base_point: Array1<f64>,
    pub coordinates: String,
    pub reference: isize,
    pub training_table_kind: String,
    pub curvature: Option<ResponseGeometryCurvature>,
}

/// Core summary of a fitted response-geometry model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseGeometrySummary {
    pub model_class: String,
    pub metadata: ResponseGeometryMetadata,
    pub tangent_dimension: usize,
    pub shared_smoothing: bool,
    pub reml_score: f64,
    pub lambdas: Array1<f64>,
    pub edf_by_penalty: Array1<f64>,
    pub edf_total: f64,
    pub sigma2: f64,
    pub template_formula: String,
    pub template_family: String,
}

/// A complete typed response-geometry model archive.
///
/// The scalar template is the native [`FittedModel`] used to reconstruct the
/// formula design for new data.  The joint tangent coefficients and REML
/// diagnostics stay in [`SharedTangentRemlFit`]; no opaque Python bytes,
/// base64, or Python-side matrix multiplication are part of this format.
#[derive(Clone, Serialize, Deserialize)]
pub struct ResponseGeometryModel {
    pub version: u32,
    pub template_model: FittedModel,
    pub metadata: ResponseGeometryMetadata,
    pub shared_tangent_fit: SharedTangentRemlFit,
}

#[derive(Debug, thiserror::Error)]
pub enum ResponseGeometryModelError {
    #[error("invalid response-geometry metadata: {0}")]
    InvalidMetadata(String),
    #[error("invalid response-geometry template model: {0}")]
    InvalidTemplate(String),
    #[error("response-geometry archive serialization failed: {0}")]
    Serialization(String),
}

impl ResponseGeometryModel {
    pub fn new(
        template_model: FittedModel,
        metadata: ResponseGeometryMetadata,
        shared_tangent_fit: SharedTangentRemlFit,
    ) -> Result<Self, ResponseGeometryModelError> {
        let model = Self {
            version: RESPONSE_GEOMETRY_MODEL_VERSION,
            template_model,
            metadata,
            shared_tangent_fit,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ResponseGeometryModelError> {
        if self.version != RESPONSE_GEOMETRY_MODEL_VERSION {
            return Err(ResponseGeometryModelError::InvalidMetadata(format!(
                "archive version {} does not match required version {}",
                self.version, RESPONSE_GEOMETRY_MODEL_VERSION
            )));
        }
        if self.metadata.response_geometry.trim().is_empty()
            || self.metadata.coordinates.trim().is_empty()
        {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "geometry and coordinate-chart labels must not be empty".to_string(),
            ));
        }
        if self.metadata.response_columns.is_empty()
            || self
                .metadata
                .response_columns
                .iter()
                .any(|column| column.trim().is_empty())
        {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "response columns must be non-empty names".to_string(),
            ));
        }
        let unique: std::collections::HashSet<&str> = self
            .metadata
            .response_columns
            .iter()
            .map(String::as_str)
            .collect();
        if unique.len() != self.metadata.response_columns.len() {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "response columns must be unique".to_string(),
            ));
        }
        if self.metadata.base_point.is_empty()
            || self
                .metadata
                .base_point
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "base point must be non-empty and finite".to_string(),
            ));
        }
        if self.metadata.training_table_kind.trim().is_empty() {
            return Err(ResponseGeometryModelError::InvalidMetadata(
                "training table kind must be non-empty".to_string(),
            ));
        }
        if let Some(curvature) = self.metadata.curvature.as_ref() {
            curvature.validate()?;
        }
        validate_archived_tangent_fit(&self.shared_tangent_fit)?;
        self.template_model
            .validate_for_persistence()
            .map_err(|error| ResponseGeometryModelError::InvalidTemplate(error.to_string()))?;
        self.template_model
            .validate_numeric_finiteness()
            .map_err(|error| ResponseGeometryModelError::InvalidTemplate(error.to_string()))?;
        Ok(())
    }

    /// Serialize the complete typed archive as UTF-8 JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ResponseGeometryModelError> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| ResponseGeometryModelError::Serialization(error.to_string()))
    }

    /// Restore and validate a complete typed archive from UTF-8 JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ResponseGeometryModelError> {
        let model: Self = serde_json::from_slice(bytes)
            .map_err(|error| ResponseGeometryModelError::Serialization(error.to_string()))?;
        model.validate()?;
        Ok(model)
    }

    pub fn metadata(&self) -> ResponseGeometryMetadata {
        self.metadata.clone()
    }

    pub fn summary(&self) -> ResponseGeometrySummary {
        let payload = self.template_model.payload();
        ResponseGeometrySummary {
            model_class: "response-geometry".to_string(),
            metadata: self.metadata.clone(),
            tangent_dimension: self.shared_tangent_fit.coefficients.ncols(),
            shared_smoothing: true,
            reml_score: self.shared_tangent_fit.reml_score,
            lambdas: self.shared_tangent_fit.lambdas.clone(),
            edf_by_penalty: self.shared_tangent_fit.edf_by_penalty.clone(),
            edf_total: self.shared_tangent_fit.edf_total,
            sigma2: self.shared_tangent_fit.sigma2,
            template_formula: payload.formula.clone(),
            template_family: payload.family.clone(),
        }
    }

    /// Predict tangent coordinates from the supplied already-materialized
    /// template design. Geometry exp-map dispatch intentionally remains in the
    /// geometry/FFI layer.
    pub fn predict_tangent(&self, design: &DesignMatrix) -> Result<Array2<f64>, EstimationError> {
        self.shared_tangent_fit.predict(design)
    }
}

#[derive(Clone, Debug)]
struct PreparedPenalty {
    output_slot: usize,
    column_start: usize,
    local: Array2<f64>,
    rank: usize,
}

#[derive(Clone, Debug)]
enum SufficientStatistics {
    Isotropic {
        gram: Array2<f64>,
        cross: Array2<f64>,
    },
    Fisher {
        gram: Array2<f64>,
        cross: Array1<f64>,
    },
}

#[derive(Clone, Debug)]
struct PreparedSharedTangent {
    design: DesignMatrix,
    response: Array2<f64>,
    weights: Array1<f64>,
    fisher_metric: Option<Array3<f64>>,
    n_observations: usize,
    n_coefficients: usize,
    n_outputs: usize,
    effective_observations: usize,
    output_penalty_slots: usize,
    penalties: Vec<PreparedPenalty>,
    statistics: SufficientStatistics,
}

#[derive(Debug)]
struct Evaluation {
    cost: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
    coefficients: Array2<f64>,
    profiled_deviance: f64,
    penalty_traces: Array1<f64>,
}

#[derive(Debug)]
struct PenaltySpectrum {
    rank: usize,
    log_pseudo_determinant: f64,
    pseudo_inverse: Array2<f64>,
}

/// Fit a shared-smoothing multi-output Gaussian model by exact profiled REML.
pub fn fit_shared_tangent_reml(
    mut request: SharedTangentRemlRequest,
) -> Result<SharedTangentRemlFit, EstimationError> {
    let requested_penalty_count = request.penalties.len();
    let initial_log_lambdas = request.initial_log_lambdas.take();
    let prepared = PreparedSharedTangent::from_request(request)?;
    let n_outer = prepared.penalties.len();
    if let Some(initial) = initial_log_lambdas.as_ref() {
        if initial.len() != requested_penalty_count {
            return Err(invalid(format!(
                "initial_log_lambdas has length {}, expected {}",
                initial.len(),
                requested_penalty_count
            )));
        }
        if initial.iter().any(|value| !value.is_finite()) {
            return Err(invalid("initial_log_lambdas must be finite"));
        }
    }
    let (rho, outer_iterations, certificate) = if n_outer == 0 {
        // A parametric model has no smoothing estimand. Its empty analytic
        // score is exactly stationary and its empty Hessian is PSD by
        // convention; record that direct certificate instead of routing a
        // zero-dimensional problem through smoothing-parameter seeding.
        (
            Array1::<f64>::zeros(0),
            0,
            OuterCriterionCertificate {
                stationarity:
                    gam_solve::rho_optimizer::OuterStationarityCertificate::AnalyticGradient {
                        grad_norm: 0.0,
                        projected_grad_norm: 0.0,
                        bound: 0.0,
                    },
                hessian_psd: Some(true),
                lambdas_railed: Vec::new(),
            },
        )
    } else {
        let mut problem = OuterProblem::new(n_outer)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Dense)
            .with_disable_fixed_point(true)
            .with_continuation_prewarm(false)
            .with_objective_scale(Some(
                prepared
                    .effective_observations
                    .checked_mul(prepared.n_outputs)
                    .ok_or_else(|| invalid("effective observation count overflow"))?
                    as f64,
            ));
        if let Some(initial) = initial_log_lambdas.as_ref() {
            problem = problem.with_initial_rho(Array1::from_iter(
                prepared
                    .penalties
                    .iter()
                    .map(|penalty| initial[penalty.output_slot]),
            ));
        }
        let mut objective = SharedTangentObjective {
            prepared: &prepared,
        };
        let outer = problem.run(&mut objective, FIT_CONTEXT)?;
        let certificate = outer
            .criterion_certificate
            .clone()
            .filter(OuterCriterionCertificate::certifies)
            .ok_or_else(|| EstimationError::RemlDidNotConverge {
                context: FIT_CONTEXT.to_string(),
                reason: "outer runner returned without a valid analytic certificate".to_string(),
                iterations: outer.iterations,
                final_value: outer.final_value,
                projected_grad_norm: outer
                    .criterion_certificate
                    .as_ref()
                    .map(|value| value.stationarity.projected_norm()),
                stationarity_bound: outer
                    .criterion_certificate
                    .as_ref()
                    .map_or(0.0, |value| value.stationarity.bound()),
                rho_checkpoint: outer.rho.to_vec(),
            })?;
        (outer.rho, outer.iterations, certificate)
    };

    let evaluation = prepared.evaluate(&rho)?;
    let fitted = predict_from_coefficients(&prepared.design, &evaluation.coefficients)?;
    let mut lambdas = Array1::<f64>::zeros(prepared.output_penalty_slots);
    let mut edf_by_penalty = Array1::<f64>::zeros(prepared.output_penalty_slots);
    for (active_index, penalty) in prepared.penalties.iter().enumerate() {
        lambdas[penalty.output_slot] = rho[active_index].exp();
        let upper = (penalty.rank * prepared.n_outputs) as f64;
        let raw = upper - evaluation.penalty_traces[active_index];
        edf_by_penalty[penalty.output_slot] =
            bounded_roundoff_value(raw, 0.0, upper, "per-penalty effective degrees of freedom")?;
    }
    let total_coefficients = prepared
        .n_coefficients
        .checked_mul(prepared.n_outputs)
        .ok_or_else(|| invalid("coefficient dimension overflow"))?
        as f64;
    let edf_total = bounded_roundoff_value(
        total_coefficients - evaluation.penalty_traces.sum(),
        0.0,
        total_coefficients,
        "total effective degrees of freedom",
    )?;
    let effective_joint_rows = prepared
        .effective_observations
        .checked_mul(prepared.n_outputs)
        .ok_or_else(|| invalid("effective joint row count overflow"))?
        as f64;
    let residual_df = effective_joint_rows - edf_total;
    if !(residual_df.is_finite() && residual_df > 0.0) {
        return Err(invalid(format!(
            "residual scale requires positive n*D-edf; got {effective_joint_rows} - {edf_total} = {residual_df}"
        )));
    }

    Ok(SharedTangentRemlFit {
        coefficients: evaluation.coefficients,
        fitted,
        sigma2: evaluation.profiled_deviance / residual_df,
        lambdas,
        edf_by_penalty,
        edf_total,
        reml_score: evaluation.cost,
        n_observations: prepared.n_observations,
        n_outputs: prepared.n_outputs,
        outer_iterations,
        outer_certificate: certificate,
    })
}

struct SharedTangentObjective<'a> {
    prepared: &'a PreparedSharedTangent,
}

impl OuterObjective for SharedTangentObjective<'_> {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Dense,
            n_params: self.prepared.penalties.len(),
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.prepared
            .evaluate(rho)
            .map(|evaluation| evaluation.cost)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let evaluation = self.prepared.evaluate(rho)?;
        Ok(OuterEval {
            cost: evaluation.cost,
            gradient: evaluation.gradient,
            hessian: HessianValue::Dense(evaluation.hessian),
            inner_beta_hint: None,
        })
    }

    fn reset(&mut self) {}

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // No warm-start slot to fill, but a non-finite seed is a caller error
        // worth surfacing rather than silently discarding.
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "seed_inner_state received a non-finite β warm-start vector",
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

impl PreparedSharedTangent {
    fn from_request(request: SharedTangentRemlRequest) -> Result<Self, EstimationError> {
        let SharedTangentRemlRequest {
            design,
            response,
            weights,
            fisher_metric,
            penalties: requested_penalties,
            initial_log_lambdas: _,
        } = request;
        let n = design.nrows();
        let k = design.ncols();
        let (response_rows, d) = response.dim();
        if n == 0 || k == 0 || d == 0 {
            return Err(invalid(format!(
                "shared-tangent REML requires non-empty dimensions; got N={n}, K={k}, D={d}"
            )));
        }
        if response_rows != n {
            return Err(invalid(format!(
                "response rows {response_rows} do not match design rows {n}"
            )));
        }
        if weights.len() != n {
            return Err(invalid(format!(
                "weight length {} does not match design rows {n}",
                weights.len()
            )));
        }
        if response.iter().any(|value| !value.is_finite()) {
            return Err(invalid("response must contain only finite values"));
        }
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid("weights must be finite and non-negative"));
        }
        let effective_observations = weights.iter().filter(|value| **value > 0.0).count();
        if effective_observations == 0 {
            return Err(invalid(
                "at least one observation must have positive weight",
            ));
        }
        if let Some(metric) = fisher_metric.as_ref()
            && metric.dim() != (n, d, d)
        {
            return Err(invalid(format!(
                "fisher_metric shape {:?} does not match ({n}, {d}, {d})",
                metric.dim()
            )));
        }
        let fisher_metric = if let Some(metric) = fisher_metric {
            let mut validated = Array3::<f64>::zeros(metric.dim());
            for row in 0..n {
                let row_metric = validated_metric(metric.slice(s![row, .., ..]).to_owned(), row)?;
                validated.slice_mut(s![row, .., ..]).assign(&row_metric);
            }
            Some(validated)
        } else {
            None
        };

        let penalties = prepare_penalties(&requested_penalties, k)?;
        let output_penalty_slots = requested_penalties.len();
        let statistics = match fisher_metric.as_ref() {
            None => assemble_isotropic_statistics(&design, &response, &weights)?,
            Some(metric) => assemble_fisher_statistics(&design, &response, &weights, metric)?,
        };

        Ok(Self {
            design,
            response,
            weights,
            fisher_metric,
            n_observations: n,
            n_coefficients: k,
            n_outputs: d,
            effective_observations,
            output_penalty_slots,
            penalties,
            statistics,
        })
    }

    fn evaluate(&self, rho: &Array1<f64>) -> Result<Evaluation, EstimationError> {
        if rho.len() != self.penalties.len() {
            return Err(invalid(format!(
                "log-lambda length {} does not match active penalty count {}",
                rho.len(),
                self.penalties.len()
            )));
        }
        if rho.iter().any(|value| !value.is_finite()) {
            return Err(invalid("log-lambdas must be finite"));
        }
        match &self.statistics {
            SufficientStatistics::Isotropic { gram, cross } => {
                self.evaluate_isotropic(rho, gram, cross)
            }
            SufficientStatistics::Fisher { gram, cross } => self.evaluate_fisher(rho, gram, cross),
        }
    }

    fn evaluate_isotropic(
        &self,
        rho: &Array1<f64>,
        gram: &Array2<f64>,
        cross: &Array2<f64>,
    ) -> Result<Evaluation, EstimationError> {
        let d = self.n_outputs;
        let (penalty, lambdas) = self.combined_penalty(rho)?;
        let spectrum = penalty_spectrum(&penalty, "combined shared-tangent penalty")?;
        let mut penalized = gram.clone();
        penalized += &penalty;
        let (inverse, log_determinant) = spd_inverse_and_logdet(&penalized)?;
        let coefficients = inverse.dot(cross);
        let profiled_deviance = self.profiled_deviance(&coefficients)?;
        let residual_degrees_of_freedom = self.residual_degrees_of_freedom(spectrum.rank)?;
        validate_profiled_deviance(profiled_deviance)?;

        let m = self.penalties.len();
        let mut penalty_traces = Array1::<f64>::zeros(m);
        let mut penalty_logdet_traces = Array1::<f64>::zeros(m);
        let mut deviance_first = Array1::<f64>::zeros(m);
        let mut penalty_beta = Vec::with_capacity(m);
        for (index, penalty_block) in self.penalties.iter().enumerate() {
            penalty_traces[index] =
                d as f64 * trace_local_base(&inverse, penalty_block, lambdas[index]);
            penalty_logdet_traces[index] = d as f64
                * trace_local_base(&spectrum.pseudo_inverse, penalty_block, lambdas[index]);
            let z = apply_local_base_matrix(penalty_block, lambdas[index], &coefficients);
            deviance_first[index] = sum_products(&coefficients, &z);
            penalty_beta.push(z);
        }

        let mut gradient = Array1::<f64>::zeros(m);
        for j in 0..m {
            gradient[j] = 0.5
                * (penalty_traces[j] - penalty_logdet_traces[j]
                    + residual_degrees_of_freedom * deviance_first[j] / profiled_deviance);
        }
        let mut hessian = Array2::<f64>::zeros((m, m));
        for j in 0..m {
            let h_sandwich = sandwich_local_base(&inverse, &self.penalties[j], lambdas[j]);
            let p_sandwich =
                sandwich_local_base(&spectrum.pseudo_inverse, &self.penalties[j], lambdas[j]);
            for kk in 0..=j {
                let h_cross = d as f64
                    * trace_sandwich_local_base(&h_sandwich, &self.penalties[kk], lambdas[kk]);
                let p_cross = d as f64
                    * trace_sandwich_local_base(&p_sandwich, &self.penalties[kk], lambdas[kk]);
                let solved_penalty_beta = inverse.dot(&penalty_beta[kk]);
                let deviance_cross = sum_products(&penalty_beta[j], &solved_penalty_beta);
                let delta = usize::from(j == kk) as f64;
                let logdet_second = delta * penalty_traces[j] - h_cross;
                let penalty_logdet_second = delta * penalty_logdet_traces[j] - p_cross;
                let deviance_second = delta * deviance_first[j] - 2.0 * deviance_cross;
                let value = 0.5
                    * (logdet_second - penalty_logdet_second
                        + residual_degrees_of_freedom
                            * (deviance_second / profiled_deviance
                                - deviance_first[j] * deviance_first[kk]
                                    / (profiled_deviance * profiled_deviance)));
                hessian[[j, kk]] = value;
                hessian[[kk, j]] = value;
            }
        }
        let cost = 0.5
            * (d as f64 * log_determinant - d as f64 * spectrum.log_pseudo_determinant
                + residual_degrees_of_freedom
                    * (1.0
                        + (2.0 * std::f64::consts::PI * profiled_deviance
                            / residual_degrees_of_freedom)
                            .ln()));
        validate_evaluation(cost, &gradient, &hessian)?;
        Ok(Evaluation {
            cost,
            gradient,
            hessian,
            coefficients,
            profiled_deviance,
            penalty_traces,
        })
    }

    fn evaluate_fisher(
        &self,
        rho: &Array1<f64>,
        gram: &Array2<f64>,
        cross: &Array1<f64>,
    ) -> Result<Evaluation, EstimationError> {
        let k = self.n_coefficients;
        let d = self.n_outputs;
        let q = k
            .checked_mul(d)
            .ok_or_else(|| invalid("joint coefficient dimension overflow"))?;
        let (penalty, lambdas) = self.combined_penalty(rho)?;
        let spectrum = penalty_spectrum(&penalty, "combined shared-tangent penalty")?;
        let mut penalized = gram.clone();
        add_base_penalty_to_joint(&mut penalized, &penalty, d);
        let (inverse, log_determinant) = spd_inverse_and_logdet(&penalized)?;
        let beta = inverse.dot(cross);
        let mut coefficients = Array2::<f64>::zeros((k, d));
        for basis in 0..k {
            for output in 0..d {
                coefficients[[basis, output]] = beta[basis * d + output];
            }
        }
        let profiled_deviance = self.profiled_deviance(&coefficients)?;
        validate_profiled_deviance(profiled_deviance)?;
        let residual_degrees_of_freedom = self.residual_degrees_of_freedom(spectrum.rank)?;

        let m = self.penalties.len();
        let mut penalty_traces = Array1::<f64>::zeros(m);
        let mut penalty_logdet_traces = Array1::<f64>::zeros(m);
        let mut deviance_first = Array1::<f64>::zeros(m);
        let mut penalty_beta = Vec::with_capacity(m);
        for (index, penalty_block) in self.penalties.iter().enumerate() {
            penalty_traces[index] = trace_local_joint(&inverse, penalty_block, lambdas[index], d);
            penalty_logdet_traces[index] = d as f64
                * trace_local_base(&spectrum.pseudo_inverse, penalty_block, lambdas[index]);
            let z = apply_local_joint_vector(penalty_block, lambdas[index], d, &beta);
            deviance_first[index] = beta.dot(&z);
            penalty_beta.push(z);
        }

        let mut gradient = Array1::<f64>::zeros(m);
        for j in 0..m {
            gradient[j] = 0.5
                * (penalty_traces[j] - penalty_logdet_traces[j]
                    + residual_degrees_of_freedom * deviance_first[j] / profiled_deviance);
        }
        let mut hessian = Array2::<f64>::zeros((m, m));
        for j in 0..m {
            let h_sandwich = sandwich_local_joint(&inverse, &self.penalties[j], lambdas[j], d);
            let p_sandwich =
                sandwich_local_base(&spectrum.pseudo_inverse, &self.penalties[j], lambdas[j]);
            for kk in 0..=j {
                let h_cross =
                    trace_sandwich_local_joint(&h_sandwich, &self.penalties[kk], lambdas[kk], d);
                let p_cross = d as f64
                    * trace_sandwich_local_base(&p_sandwich, &self.penalties[kk], lambdas[kk]);
                let solved_penalty_beta = inverse.dot(&penalty_beta[kk]);
                let deviance_cross = penalty_beta[j].dot(&solved_penalty_beta);
                let delta = usize::from(j == kk) as f64;
                let logdet_second = delta * penalty_traces[j] - h_cross;
                let penalty_logdet_second = delta * penalty_logdet_traces[j] - p_cross;
                let deviance_second = delta * deviance_first[j] - 2.0 * deviance_cross;
                let value = 0.5
                    * (logdet_second - penalty_logdet_second
                        + residual_degrees_of_freedom
                            * (deviance_second / profiled_deviance
                                - deviance_first[j] * deviance_first[kk]
                                    / (profiled_deviance * profiled_deviance)));
                hessian[[j, kk]] = value;
                hessian[[kk, j]] = value;
            }
        }
        let cost = 0.5
            * (log_determinant - d as f64 * spectrum.log_pseudo_determinant
                + residual_degrees_of_freedom
                    * (1.0
                        + (2.0 * std::f64::consts::PI * profiled_deviance
                            / residual_degrees_of_freedom)
                            .ln()));
        if inverse.dim() != (q, q) {
            return Err(invalid("internal Fisher inverse shape mismatch"));
        }
        validate_evaluation(cost, &gradient, &hessian)?;
        Ok(Evaluation {
            cost,
            gradient,
            hessian,
            coefficients,
            profiled_deviance,
            penalty_traces,
        })
    }

    /// Evaluate the fitted weighted residual quadratic directly from row
    /// chunks.  Forming it as `y'Wy - (X'Wy)' beta` catastrophically cancels
    /// on near-interpolating fits; the resulting few ulps are large relative to
    /// the residual itself and can move a flat REML optimum by many nats under
    /// an otherwise harmless rotation of the tangent frame.
    fn profiled_deviance(&self, coefficients: &Array2<f64>) -> Result<f64, EstimationError> {
        let n = self.design.nrows();
        let k = self.design.ncols();
        let d = self.response.ncols();
        if coefficients.dim() != (k, d) {
            return Err(invalid(format!(
                "shared-tangent coefficient shape {:?} does not match ({k}, {d})",
                coefficients.dim()
            )));
        }
        let mut quadratic = KahanSum::default();
        let chunk_rows = gam_linalg::utils::row_chunk_for_byte_budget(n, k);
        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let x_chunk = self
                .design
                .try_row_chunk(start..end)
                .map_err(|error| invalid(format!("failed to read design row chunk: {error}")))?;
            validate_design_chunk(&x_chunk)?;
            let fitted = x_chunk.dot(coefficients);
            for local_row in 0..x_chunk.nrows() {
                let row = start + local_row;
                let weight = self.weights[row];
                if weight == 0.0 {
                    continue;
                }
                if let Some(metric) = self.fisher_metric.as_ref() {
                    for output_a in 0..d {
                        let residual_a =
                            self.response[[row, output_a]] - fitted[[local_row, output_a]];
                        for output_b in 0..d {
                            let residual_b =
                                self.response[[row, output_b]] - fitted[[local_row, output_b]];
                            quadratic.add(
                                weight
                                    * residual_a
                                    * metric[[row, output_a, output_b]]
                                    * residual_b,
                            );
                        }
                    }
                } else {
                    for output in 0..d {
                        let residual = self.response[[row, output]] - fitted[[local_row, output]];
                        quadratic.add(weight * residual * residual);
                    }
                }
            }
        }
        Ok(quadratic.sum())
    }

    fn combined_penalty(
        &self,
        rho: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>), EstimationError> {
        let mut combined = Array2::<f64>::zeros((self.n_coefficients, self.n_coefficients));
        let mut lambdas = Array1::<f64>::zeros(rho.len());
        for (index, penalty) in self.penalties.iter().enumerate() {
            let lambda = rho[index].exp();
            if !lambda.is_finite() || lambda <= 0.0 {
                return Err(invalid(format!(
                    "log-lambda {} exponentiated to invalid value {lambda}",
                    rho[index]
                )));
            }
            lambdas[index] = lambda;
            for local_row in 0..penalty.local.nrows() {
                for local_col in 0..penalty.local.ncols() {
                    combined[[
                        penalty.column_start + local_row,
                        penalty.column_start + local_col,
                    ]] += lambda * penalty.local[[local_row, local_col]];
                }
            }
        }
        Ok((combined, lambdas))
    }

    fn residual_degrees_of_freedom(
        &self,
        combined_penalty_rank: usize,
    ) -> Result<f64, EstimationError> {
        let effective_rows = self
            .effective_observations
            .checked_mul(self.n_outputs)
            .ok_or_else(|| invalid("effective joint row count overflow"))?;
        let base_nullity = self
            .n_coefficients
            .checked_sub(combined_penalty_rank)
            .ok_or_else(|| invalid("combined penalty rank exceeds coefficient dimension"))?;
        let joint_nullity = base_nullity
            .checked_mul(self.n_outputs)
            .ok_or_else(|| invalid("joint penalty nullity overflow"))?;
        if effective_rows <= joint_nullity {
            return Err(invalid(format!(
                "REML requires more effective joint rows than unpenalized coefficients; got {effective_rows} rows and nullity {joint_nullity}"
            )));
        }
        Ok((effective_rows - joint_nullity) as f64)
    }
}

fn prepare_penalties(
    penalties: &[SharedTangentPenalty],
    n_coefficients: usize,
) -> Result<Vec<PreparedPenalty>, EstimationError> {
    let mut prepared = Vec::with_capacity(penalties.len());
    for (slot, penalty) in penalties.iter().enumerate() {
        let q = penalty.matrix.nrows();
        if q != penalty.matrix.ncols() {
            return Err(invalid(format!(
                "penalty {slot} must be square; got {}x{}",
                penalty.matrix.nrows(),
                penalty.matrix.ncols()
            )));
        }
        let end = penalty
            .column_start
            .checked_add(q)
            .ok_or_else(|| invalid(format!("penalty {slot} column range overflow")))?;
        if end > n_coefficients {
            return Err(invalid(format!(
                "penalty {slot} column range {}..{end} exceeds design width {n_coefficients}",
                penalty.column_start
            )));
        }
        if penalty.matrix.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "penalty {slot} contains non-finite values"
            )));
        }
        let local = symmetric_average(&penalty.matrix);
        let spectrum = penalty_spectrum(&local, &format!("shared-tangent penalty {slot}"))?;
        if spectrum.rank == 0 {
            continue;
        }
        prepared.push(PreparedPenalty {
            output_slot: slot,
            column_start: penalty.column_start,
            local,
            rank: spectrum.rank,
        });
    }
    Ok(prepared)
}

fn assemble_isotropic_statistics(
    design: &DesignMatrix,
    response: &Array2<f64>,
    weights: &Array1<f64>,
) -> Result<SufficientStatistics, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let d = response.ncols();
    let mut gram = Array2::<f64>::zeros((k, k));
    let mut cross = Array2::<f64>::zeros((k, d));
    let chunk_rows = gam_linalg::utils::row_chunk_for_byte_budget(n, k);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let x_chunk = design
            .try_row_chunk(start..end)
            .map_err(|error| invalid(format!("failed to read design row chunk: {error}")))?;
        validate_design_chunk(&x_chunk)?;
        let weight_chunk = weights.slice(s![start..end]);
        let response_chunk = response.slice(s![start..end, ..]);
        gram += &fast_xt_diag_x(&x_chunk, &weight_chunk);
        cross += &fast_xt_diag_y(&x_chunk, &weight_chunk, &response_chunk);
    }
    Ok(SufficientStatistics::Isotropic { gram, cross })
}

fn assemble_fisher_statistics(
    design: &DesignMatrix,
    response: &Array2<f64>,
    weights: &Array1<f64>,
    fisher_metric: &Array3<f64>,
) -> Result<SufficientStatistics, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let d = response.ncols();
    let q = k
        .checked_mul(d)
        .ok_or_else(|| invalid("joint coefficient dimension overflow"))?;
    let mut gram = Array2::<f64>::zeros((q, q));
    let mut cross = Array1::<f64>::zeros(q);
    let chunk_rows = gam_linalg::utils::row_chunk_for_byte_budget(n, k);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let x_chunk = design
            .try_row_chunk(start..end)
            .map_err(|error| invalid(format!("failed to read design row chunk: {error}")))?;
        validate_design_chunk(&x_chunk)?;
        for local_row in 0..x_chunk.nrows() {
            let row = start + local_row;
            let metric = fisher_metric.slice(s![row, .., ..]);
            let y = response.row(row);
            let metric_y = metric.dot(&y);
            let weight = weights[row];
            for basis_a in 0..k {
                let x_a = x_chunk[[local_row, basis_a]];
                for output in 0..d {
                    cross[basis_a * d + output] += weight * x_a * metric_y[output];
                }
                for basis_b in 0..k {
                    let scale = weight * x_a * x_chunk[[local_row, basis_b]];
                    if scale == 0.0 {
                        continue;
                    }
                    for output_a in 0..d {
                        for output_b in 0..d {
                            gram[[basis_a * d + output_a, basis_b * d + output_b]] +=
                                scale * metric[[output_a, output_b]];
                        }
                    }
                }
            }
        }
    }
    Ok(SufficientStatistics::Fisher { gram, cross })
}

fn validated_metric(mut metric: Array2<f64>, row: usize) -> Result<Array2<f64>, EstimationError> {
    if metric.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "fisher_metric row {row} contains non-finite values"
        )));
    }
    let scale = metric
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let tolerance = f64::EPSILON.sqrt() * metric.nrows().max(1) as f64 * scale;
    for a in 0..metric.nrows() {
        for b in (a + 1)..metric.ncols() {
            if (metric[[a, b]] - metric[[b, a]]).abs() > tolerance {
                return Err(invalid(format!(
                    "fisher_metric row {row} is not symmetric at ({a}, {b})"
                )));
            }
            let average = 0.5 * (metric[[a, b]] + metric[[b, a]]);
            metric[[a, b]] = average;
            metric[[b, a]] = average;
        }
    }
    metric.cholesky(Side::Lower).map_err(|error| {
        invalid(format!(
            "fisher_metric row {row} must be positive definite: {error}"
        ))
    })?;
    Ok(metric)
}

fn penalty_spectrum(
    penalty: &Array2<f64>,
    context: &str,
) -> Result<PenaltySpectrum, EstimationError> {
    let (eigenvalues, eigenvectors) = penalty
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    let scale = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let tolerance = f64::EPSILON.sqrt() * eigenvalues.len().max(1) as f64 * scale;
    let mut rank = 0usize;
    let mut log_pseudo_determinant = 0.0;
    let mut pseudo_inverse = Array2::<f64>::zeros(penalty.dim());
    for (index, &value) in eigenvalues.iter().enumerate() {
        if !value.is_finite() {
            return Err(EstimationError::PenaltySpectrumNonFinite {
                context: context.to_string(),
                index,
                value,
            });
        }
        if value < -tolerance {
            return Err(EstimationError::PenaltySpectrumIndefinite {
                context: context.to_string(),
                index,
                value,
                tolerance,
                scale,
            });
        }
        if value <= tolerance {
            continue;
        }
        rank += 1;
        log_pseudo_determinant += value.ln();
        for row in 0..penalty.nrows() {
            for col in 0..penalty.ncols() {
                pseudo_inverse[[row, col]] +=
                    eigenvectors[[row, index]] * eigenvectors[[col, index]] / value;
            }
        }
    }
    Ok(PenaltySpectrum {
        rank,
        log_pseudo_determinant,
        pseudo_inverse,
    })
}

fn spd_inverse_and_logdet(matrix: &Array2<f64>) -> Result<(Array2<f64>, f64), EstimationError> {
    let factor =
        matrix
            .cholesky(Side::Lower)
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let diagonal = factor.diag();
    if diagonal
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    let log_determinant = 2.0 * diagonal.iter().map(|value| value.ln()).sum::<f64>();
    let identity = Array2::<f64>::eye(matrix.nrows());
    let inverse = factor.solve_mat(&identity);
    if !log_determinant.is_finite() || inverse.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    Ok((inverse, log_determinant))
}

fn trace_local_base(inverse: &Array2<f64>, penalty: &PreparedPenalty, lambda: f64) -> f64 {
    let mut trace = 0.0;
    for row in 0..penalty.local.nrows() {
        for col in 0..penalty.local.ncols() {
            trace += lambda
                * penalty.local[[row, col]]
                * inverse[[penalty.column_start + col, penalty.column_start + row]];
        }
    }
    trace
}

fn trace_local_joint(
    inverse: &Array2<f64>,
    penalty: &PreparedPenalty,
    lambda: f64,
    n_outputs: usize,
) -> f64 {
    let mut trace = 0.0;
    for output in 0..n_outputs {
        for row in 0..penalty.local.nrows() {
            for col in 0..penalty.local.ncols() {
                trace += lambda
                    * penalty.local[[row, col]]
                    * inverse[[
                        (penalty.column_start + col) * n_outputs + output,
                        (penalty.column_start + row) * n_outputs + output,
                    ]];
            }
        }
    }
    trace
}

fn apply_local_base_matrix(
    penalty: &PreparedPenalty,
    lambda: f64,
    matrix: &Array2<f64>,
) -> Array2<f64> {
    let mut output = Array2::<f64>::zeros(matrix.dim());
    for row in 0..penalty.local.nrows() {
        for col in 0..penalty.local.ncols() {
            let value = lambda * penalty.local[[row, col]];
            for output_index in 0..matrix.ncols() {
                output[[penalty.column_start + row, output_index]] +=
                    value * matrix[[penalty.column_start + col, output_index]];
            }
        }
    }
    output
}

fn apply_local_joint_vector(
    penalty: &PreparedPenalty,
    lambda: f64,
    n_outputs: usize,
    vector: &Array1<f64>,
) -> Array1<f64> {
    let mut output = Array1::<f64>::zeros(vector.len());
    for output_index in 0..n_outputs {
        for row in 0..penalty.local.nrows() {
            for col in 0..penalty.local.ncols() {
                output[(penalty.column_start + row) * n_outputs + output_index] += lambda
                    * penalty.local[[row, col]]
                    * vector[(penalty.column_start + col) * n_outputs + output_index];
            }
        }
    }
    output
}

fn sandwich_local_base(
    inverse: &Array2<f64>,
    penalty: &PreparedPenalty,
    lambda: f64,
) -> Array2<f64> {
    let dimension = inverse.nrows();
    let mut result = Array2::<f64>::zeros((dimension, dimension));
    for local_row in 0..penalty.local.nrows() {
        let global_row = penalty.column_start + local_row;
        for local_col in 0..penalty.local.ncols() {
            let value = lambda * penalty.local[[local_row, local_col]];
            if value == 0.0 {
                continue;
            }
            let global_col = penalty.column_start + local_col;
            for row in 0..dimension {
                let left = inverse[[row, global_row]] * value;
                for col in 0..dimension {
                    result[[row, col]] += left * inverse[[global_col, col]];
                }
            }
        }
    }
    result
}

fn sandwich_local_joint(
    inverse: &Array2<f64>,
    penalty: &PreparedPenalty,
    lambda: f64,
    n_outputs: usize,
) -> Array2<f64> {
    let dimension = inverse.nrows();
    let mut result = Array2::<f64>::zeros((dimension, dimension));
    for output in 0..n_outputs {
        for local_row in 0..penalty.local.nrows() {
            let global_row = (penalty.column_start + local_row) * n_outputs + output;
            for local_col in 0..penalty.local.ncols() {
                let value = lambda * penalty.local[[local_row, local_col]];
                if value == 0.0 {
                    continue;
                }
                let global_col = (penalty.column_start + local_col) * n_outputs + output;
                for row in 0..dimension {
                    let left = inverse[[row, global_row]] * value;
                    for col in 0..dimension {
                        result[[row, col]] += left * inverse[[global_col, col]];
                    }
                }
            }
        }
    }
    result
}

fn trace_sandwich_local_base(
    sandwich: &Array2<f64>,
    penalty: &PreparedPenalty,
    lambda: f64,
) -> f64 {
    let mut trace = 0.0;
    for row in 0..penalty.local.nrows() {
        for col in 0..penalty.local.ncols() {
            trace += lambda
                * penalty.local[[row, col]]
                * sandwich[[penalty.column_start + col, penalty.column_start + row]];
        }
    }
    trace
}

fn trace_sandwich_local_joint(
    sandwich: &Array2<f64>,
    penalty: &PreparedPenalty,
    lambda: f64,
    n_outputs: usize,
) -> f64 {
    let mut trace = 0.0;
    for output in 0..n_outputs {
        for row in 0..penalty.local.nrows() {
            for col in 0..penalty.local.ncols() {
                trace += lambda
                    * penalty.local[[row, col]]
                    * sandwich[[
                        (penalty.column_start + col) * n_outputs + output,
                        (penalty.column_start + row) * n_outputs + output,
                    ]];
            }
        }
    }
    trace
}

fn add_base_penalty_to_joint(joint: &mut Array2<f64>, penalty: &Array2<f64>, n_outputs: usize) {
    for row in 0..penalty.nrows() {
        for col in 0..penalty.ncols() {
            let value = penalty[[row, col]];
            for output in 0..n_outputs {
                joint[[row * n_outputs + output, col * n_outputs + output]] += value;
            }
        }
    }
}

fn symmetric_average(matrix: &Array2<f64>) -> Array2<f64> {
    let mut output = matrix.clone();
    for row in 0..matrix.nrows() {
        for col in (row + 1)..matrix.ncols() {
            let average = 0.5 * (matrix[[row, col]] + matrix[[col, row]]);
            output[[row, col]] = average;
            output[[col, row]] = average;
        }
    }
    output
}

fn predict_from_coefficients(
    design: &DesignMatrix,
    coefficients: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    if design.ncols() != coefficients.nrows() {
        return Err(invalid(format!(
            "prediction design width {} does not match coefficient rows {}",
            design.ncols(),
            coefficients.nrows()
        )));
    }
    let mut prediction = Array2::<f64>::zeros((design.nrows(), coefficients.ncols()));
    for output in 0..coefficients.ncols() {
        let values = design.apply(&coefficients.column(output).to_owned());
        prediction.column_mut(output).assign(&values);
    }
    if prediction.iter().any(|value| !value.is_finite()) {
        return Err(invalid("prediction produced non-finite values"));
    }
    Ok(prediction)
}

fn sum_products(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn validate_design_chunk(chunk: &Array2<f64>) -> Result<(), EstimationError> {
    if chunk.iter().any(|value| !value.is_finite()) {
        return Err(invalid("design contains non-finite values"));
    }
    Ok(())
}

fn validate_profiled_deviance(value: f64) -> Result<(), EstimationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "{FIT_CONTEXT}: profiled penalized deviance must be finite and positive, got {value}"
        )));
    }
    Ok(())
}

fn validate_evaluation(
    cost: f64,
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
) -> Result<(), EstimationError> {
    if !cost.is_finite()
        || gradient.iter().any(|value| !value.is_finite())
        || hessian.iter().any(|value| !value.is_finite())
    {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "{FIT_CONTEXT}: objective evaluation produced non-finite value or derivatives"
        )));
    }
    Ok(())
}

fn validate_archived_tangent_fit(
    fit: &SharedTangentRemlFit,
) -> Result<(), ResponseGeometryModelError> {
    if fit.n_observations == 0
        || fit.n_outputs == 0
        || fit.coefficients.nrows() == 0
        || fit.coefficients.ncols() != fit.n_outputs
        || fit.fitted.dim() != (fit.n_observations, fit.n_outputs)
    {
        return Err(ResponseGeometryModelError::InvalidMetadata(
            "shared tangent fit has inconsistent dimensions".to_string(),
        ));
    }
    if fit.lambdas.len() != fit.edf_by_penalty.len() {
        return Err(ResponseGeometryModelError::InvalidMetadata(
            "shared tangent lambda and EDF vectors are misaligned".to_string(),
        ));
    }
    if fit.coefficients.iter().any(|value| !value.is_finite())
        || fit.fitted.iter().any(|value| !value.is_finite())
        || fit
            .lambdas
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || fit
            .edf_by_penalty
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || !fit.sigma2.is_finite()
        || fit.sigma2 <= 0.0
        || !fit.edf_total.is_finite()
        || fit.edf_total < 0.0
        || !fit.reml_score.is_finite()
    {
        return Err(ResponseGeometryModelError::InvalidMetadata(
            "shared tangent fit contains invalid numerical values".to_string(),
        ));
    }
    if !fit.outer_certificate.certifies() {
        return Err(ResponseGeometryModelError::InvalidMetadata(
            "shared tangent fit lacks a valid convergence certificate".to_string(),
        ));
    }
    Ok(())
}

fn bounded_roundoff_value(
    value: f64,
    lower: f64,
    upper: f64,
    context: &str,
) -> Result<f64, EstimationError> {
    let tolerance = f64::EPSILON.sqrt() * upper.abs().max(1.0);
    if !value.is_finite() || value < lower - tolerance || value > upper + tolerance {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "{FIT_CONTEXT}: {context} {value} lies outside [{lower}, {upper}] beyond roundoff"
        )));
    }
    Ok(value.clamp(lower, upper))
}

fn invalid(message: impl Into<String>) -> EstimationError {
    EstimationError::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::test_support::no_densify_design;
    use ndarray::{Array3, array};

    fn fixture_request(fisher_metric: Option<Array3<f64>>) -> SharedTangentRemlRequest {
        let design = array![
            [1.0, -1.0, 0.5],
            [1.0, -0.5, -0.2],
            [1.0, 0.0, 0.3],
            [1.0, 0.5, 0.8],
            [1.0, 1.0, -0.4],
            [1.0, 1.5, 0.1]
        ];
        let response = array![
            [-0.7, 0.4],
            [-0.1, 0.1],
            [0.2, -0.3],
            [0.8, -0.2],
            [1.1, 0.5],
            [1.7, 0.2]
        ];
        let penalties = vec![
            SharedTangentPenalty::new(1, array![[1.0, 0.0], [0.0, 0.0]]),
            SharedTangentPenalty::new(1, array![[0.0, 0.0], [0.0, 1.0]]),
        ];
        SharedTangentRemlRequest::new(
            no_densify_design(design),
            response,
            array![1.0, 0.8, 1.2, 1.0, 0.9, 1.1],
            fisher_metric,
            penalties,
        )
    }

    #[test]
    fn operator_backed_isotropic_path_matches_streamed_identity_fisher_path() {
        let isotropic_request = fixture_request(None);
        let n = isotropic_request.response.nrows();
        let d = isotropic_request.response.ncols();
        let mut identity_metric = Array3::<f64>::zeros((n, d, d));
        for row in 0..n {
            for output in 0..d {
                identity_metric[[row, output, output]] = 1.0;
            }
        }
        let fisher_request = fixture_request(Some(identity_metric));
        let isotropic = PreparedSharedTangent::from_request(isotropic_request)
            .expect("prepare isotropic without densifying");
        let fisher = PreparedSharedTangent::from_request(fisher_request)
            .expect("prepare Fisher without densifying");
        let rho = array![-0.4, 0.7];
        let left = isotropic.evaluate(&rho).expect("isotropic eval");
        let right = fisher.evaluate(&rho).expect("Fisher eval");
        assert_close(left.cost, right.cost, 2.0e-11);
        assert_array1_close(&left.gradient, &right.gradient, 2.0e-10);
        assert_array2_close(&left.hessian, &right.hessian, 2.0e-9);
        assert_array2_close(&left.coefficients, &right.coefficients, 2.0e-11);
    }

    #[test]
    fn analytic_gradient_and_hessian_match_test_only_finite_differences() {
        let request = fixture_request(None);
        let prepared = PreparedSharedTangent::from_request(request).expect("prepare");
        let rho = array![-0.2, 0.35];
        let exact = prepared.evaluate(&rho).expect("exact eval");
        let step = f64::EPSILON.cbrt();
        for j in 0..rho.len() {
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            plus[j] += step;
            minus[j] -= step;
            let plus_eval = prepared.evaluate(&plus).expect("plus eval");
            let minus_eval = prepared.evaluate(&minus).expect("minus eval");
            let gradient_fd = (plus_eval.cost - minus_eval.cost) / (2.0 * step);
            assert_close(exact.gradient[j], gradient_fd, 2.0e-6);
            for k in 0..rho.len() {
                let hessian_fd = (plus_eval.gradient[k] - minus_eval.gradient[k]) / (2.0 * step);
                assert_close(exact.hessian[[k, j]], hessian_fd, 3.0e-6);
            }
        }
    }

    #[test]
    fn streamed_varying_fisher_statistics_match_explicit_joint_oracle() {
        let base = fixture_request(None);
        let n = base.response.nrows();
        let d = base.response.ncols();
        let mut metric = Array3::<f64>::zeros((n, d, d));
        for row in 0..n {
            let off = 0.04 * (row as f64 + 1.0);
            metric[[row, 0, 0]] = 1.2 + 0.1 * row as f64;
            metric[[row, 0, 1]] = off;
            metric[[row, 1, 0]] = off;
            metric[[row, 1, 1]] = 0.9 + 0.05 * row as f64;
        }
        let request = fixture_request(Some(metric.clone()));
        let prepared =
            PreparedSharedTangent::from_request(request.clone()).expect("prepare Fisher");
        let SufficientStatistics::Fisher { gram, cross } = &prepared.statistics else {
            panic!("expected Fisher statistics")
        };
        let x = base.design.try_row_chunk(0..n).expect("test design rows");
        let k = x.ncols();
        let q = k * d;
        let mut oracle_gram = Array2::<f64>::zeros((q, q));
        let mut oracle_cross = Array1::<f64>::zeros(q);
        let mut oracle_response = 0.0;
        for row in 0..n {
            for a in 0..k {
                for o in 0..d {
                    let ao = a * d + o;
                    for p in 0..d {
                        oracle_cross[ao] += request.weights[row]
                            * x[[row, a]]
                            * metric[[row, o, p]]
                            * request.response[[row, p]];
                    }
                    for b in 0..k {
                        for p in 0..d {
                            oracle_gram[[ao, b * d + p]] += request.weights[row]
                                * x[[row, a]]
                                * x[[row, b]]
                                * metric[[row, o, p]];
                        }
                    }
                }
            }
            let y = request.response.row(row);
            oracle_response += request.weights[row] * y.dot(&metric.slice(s![row, .., ..]).dot(&y));
        }
        assert_array2_close(gram, &oracle_gram, 2.0e-12);
        assert_array1_close(cross, &oracle_cross, 2.0e-12);
        let zero_coefficients = Array2::<f64>::zeros((k, d));
        let direct_response_quadratic = prepared
            .profiled_deviance(&zero_coefficients)
            .expect("direct zero-fit quadratic");
        assert_close(direct_response_quadratic, oracle_response, 2.0e-12);
    }

    #[test]
    fn parametric_fit_is_certified_serializable_and_predicts_in_core() {
        let design = array![[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let response = array![[0.2, -0.1], [0.9, 0.4], [2.1, 0.8], [2.8, 1.4]];
        let request = SharedTangentRemlRequest::from_dense(
            design.clone(),
            response,
            Array1::ones(4),
            None,
            Vec::new(),
        );
        let fit = fit_shared_tangent_reml(request).expect("certified parametric fit");
        assert!(fit.outer_certificate.certifies());
        let prediction = fit.predict_dense(design).expect("core prediction");
        assert_array2_close(&prediction, &fit.fitted, 1.0e-12);
        let encoded = serde_json::to_string(&fit).expect("serialize fit");
        let decoded: SharedTangentRemlFit =
            serde_json::from_str(&encoded).expect("deserialize fit");
        assert_array2_close(&decoded.coefficients, &fit.coefficients, 0.0);
        assert!(decoded.outer_certificate.certifies());
    }

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= tolerance * scale,
            "{left} != {right} within relative tolerance {tolerance}"
        );
    }

    fn assert_array1_close(left: &Array1<f64>, right: &Array1<f64>, tolerance: f64) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right.iter()) {
            assert_close(*left, *right, tolerance);
        }
    }

    fn assert_array2_close(left: &Array2<f64>, right: &Array2<f64>, tolerance: f64) {
        assert_eq!(left.dim(), right.dim());
        for (left, right) in left.iter().zip(right.iter()) {
            assert_close(*left, *right, tolerance);
        }
    }
}
