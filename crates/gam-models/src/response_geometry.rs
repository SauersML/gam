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
//! only `K x K` square-root statistics. Varying Fisher metrics stream exact joint
//! sufficient statistics into a `(K D) x (K D)` Gram matrix whose storage is
//! independent of `N`.

use faer::Side;
use gam_linalg::faer_ndarray::{FaerArrayView, FaerCholesky, FaerEigh, FaerQr, array2_to_matmut};
use gam_linalg::matrix::{DesignMatrix, LinearOperator};
use gam_linalg::utils::KahanSum;
// `DeclaredHessianForm`/`Derivative` originate in `gam_problem` and are only
// re-exported privately inside `gam_solve::rho_optimizer`; import them from the
// canonical source, matching every other `gam-models` outer-objective site.
use gam_problem::{DeclaredHessianForm, Derivative, StationarityStandard};
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
}

#[derive(Clone, Debug)]
struct PreparedPenalty {
    output_slot: usize,
    column_start: usize,
    local: Array2<f64>,
    root: Array2<f64>,
    rank: usize,
}

#[derive(Clone, Debug)]
enum SufficientStatistics {
    Isotropic {
        root: Array2<f64>,
        projected_response: Array2<f64>,
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
    /// The union of the unscaled penalty ranges. Positive smoothing strengths
    /// change curvature within this space, never its dimension.
    penalty_range: Array2<f64>,
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
    lambdas: Array1<f64>,
    /// `rank(Σ_j λ_j S_j)` in the PER-OUTPUT coefficient basis, carried out of
    /// the evaluation that measured it. The fit boundary needs it to state the
    /// joint penalty nullity `mp = p − rank(ΣS)` — the effective dimension no
    /// amount of smoothing can remove — and re-deriving it there would be a
    /// second spectral decision about the same matrix.
    combined_penalty_rank: usize,
}

#[derive(Debug)]
struct PenaltySpectrum {
    rank: usize,
    log_pseudo_determinant: f64,
    traces: Array1<f64>,
    cross_traces: Array2<f64>,
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
                        // A parametric model has no smoothing estimand, so the
                        // empty score is stationary by construction rather than
                        // by clearing a band; `bound: 0.0` is a formality and
                        // borrowing a gradient rung for it would name a
                        // comparison that never ran (#2530).
                        rung: gam_problem::StationarityRung::EMPTY_ESTIMAND.into(),
                    },
                // A parametric model has no smoothing estimand, so there is
                // no outer Hessian and never was — `Some(true)` claimed a
                // measurement that never ran, the second-order twin of the
                // rung mistake the comment above already avoids (#2561).
                curvature: gam_solve::rho_optimizer::CurvatureEvidence::NoEstimand,
                lambdas_railed: Vec::new(),
                railed_facts: Vec::new(),
                curvature_floor: None,
            },
        )
    } else {
        let mut problem = OuterProblem::new(n_outer)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Dense)
            .with_disable_fixed_point(true)
            // The closed-form QR/root evaluation resolves the per-output
            // smoothing score to the floating-point floor. The generic outer
            // band also serves inexact inner solves and can stop equivalent
            // response frames at distinguishable coefficient maps. State this
            // exact engine's accuracy requirement before search/certification.
            .with_required_projected_gradient_norm(Some(
                f64::EPSILON.sqrt() * prepared.n_outputs as f64,
            ))
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
                // The refusal predicate here is "the runner returned no
                // certificate that certifies" — an existence check, not a
                // stationarity comparison. Reporting the certificate's own
                // bound (or `0.0` when there is no certificate at all) beside
                // the words "against stationarity bound" named a comparison
                // this route never made (#2458/#2465).
                stationarity_standard: StationarityStandard::NoComparison,
                rho_checkpoint: outer.rho.to_vec(),
            })?;
        (outer.rho, outer.iterations, certificate)
    };

    let evaluation = prepared.evaluate(&rho)?;
    let fitted = predict_from_coefficients(&prepared.design, &evaluation.coefficients)?;
    // The EDF accounting — which ceiling a per-block trace is admitted against,
    // what a non-finite trace resolves to, what `edf_by_block` is measured
    // against, and what floor `edf_total` may not fall below — is the SHARED one
    // (`gam_solve::estimate::penalized_edf_bundle`, issue #2470). This route was
    // the last one keeping its own, and it differed on every axis that matters:
    //
    // * **Floor.** It clamped `edf_total` to `[0, p]`. The attainable minimum is
    //   the joint penalty nullity `mp = p − rank(Σ_j λ_j S_j)`: those directions
    //   are unpenalized, so no amount of smoothing removes them. A `[0, p]`
    //   clamp lets a noisy trace publish an effective dimension below the
    //   mathematically possible one, and `σ̂² = D/(rows − edf)` and every SE off
    //   this path inherit it silently.
    // * **Saturation.** A redundant block driven to the λ ceiling can overflow
    //   the raw product `λ_k·tr` to `+∞` on a ridge-stabilized system even
    //   though the true value is exactly `rank_k` (#1379). The local
    //   `bounded_roundoff_value` refuses any non-finite value, so this route
    //   FAILED THE WHOLE FIT on a case the shared accounting resolves to the
    //   saturated bound.
    // * **Summation.** `edf_total` is a difference of two like-sized quantities,
    //   so naive `.sum()` error lands directly in the reported dimension; the
    //   shared path sums the admitted traces with compensated addition.
    //
    // Stating `mp` here is also what lets `collapsed_to_penalty_null_space` see
    // a fit that kept none of the penalized directions its design offered — a
    // state this route previously reported as an ordinary converged answer.
    //
    // The per-block ceiling is `rank(S_j)·D`: shared smoothing applies each
    // penalty to the same column block in every one of the `D` outputs, so its
    // rank in the joint `p = n_coefficients·D` space is `D` copies of the local
    // rank. That is the ceiling this route already used, and it is the one the
    // REML criterion prices, so it is carried over unchanged.
    let block_ranks: Vec<usize> = prepared
        .penalties
        .iter()
        .map(|penalty| penalty.rank * prepared.n_outputs)
        .collect();
    // Kept from the accounting this route used to own: a trace that is FINITE
    // and materially outside `[0, rank]` is not saturation and not roundoff — it
    // is broken linear algebra upstream, and admitting it at a bound would hide
    // that. The shared accounting deliberately clamps (a `+∞` product is the
    // ceiling case above; a NaN or `−∞` product propagates for the finiteness
    // check below to refuse), so this input check stays here rather than
    // becoming a second accounting policy.
    for (active_index, &rank) in block_ranks.iter().enumerate() {
        let raw = evaluation.penalty_traces[active_index];
        if raw.is_finite() {
            bounded_roundoff_value(raw, 0.0, rank as f64, "per-penalty penalty trace")?;
        }
    }
    let total_coefficients = prepared
        .n_coefficients
        .checked_mul(prepared.n_outputs)
        .ok_or_else(|| invalid("coefficient dimension overflow"))?;
    let joint_penalty_nullity = prepared
        .n_coefficients
        .checked_sub(evaluation.combined_penalty_rank)
        .ok_or_else(|| invalid("combined penalty rank exceeds coefficient dimension"))?
        .checked_mul(prepared.n_outputs)
        .ok_or_else(|| invalid("joint penalty nullity overflow"))?;
    let bundle = gam_solve::estimate::penalized_edf_bundle(
        evaluation
            .penalty_traces
            .as_slice()
            .ok_or_else(|| invalid("penalty traces are not contiguous"))?,
        &block_ranks,
        total_coefficients,
        joint_penalty_nullity as f64,
    );
    let mut lambdas = Array1::<f64>::zeros(prepared.output_penalty_slots);
    let mut edf_by_penalty = Array1::<f64>::zeros(prepared.output_penalty_slots);
    for (active_index, penalty) in prepared.penalties.iter().enumerate() {
        lambdas[penalty.output_slot] = evaluation.lambdas[active_index];
        edf_by_penalty[penalty.output_slot] = bundle.edf_by_block[active_index];
    }
    let edf_total = bundle.edf_total;
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
        let balanced = gam_terms::construction::balanced_penalty_sum(
            penalties.iter().map(|penalty| {
                (
                    penalty.local.view(),
                    penalty.column_start..penalty.column_start + penalty.local.nrows(),
                )
            }),
            k,
        );
        let (values, vectors) = balanced
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let tolerance = gam_terms::construction::balanced_penalty_rank_tolerance(
            values.iter().copied().fold(0.0_f64, f64::max),
        );
        let active: Vec<usize> = values
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value > tolerance).then_some(index))
            .collect();
        let penalty_range =
            Array2::from_shape_fn((k, active.len()), |(row, col)| vectors[[row, active[col]]]);
        let output_penalty_slots = requested_penalties.len();
        let statistics = match fisher_metric.as_ref() {
            None => assemble_isotropic_statistics(
                &design,
                &response,
                &weights,
                gam_linalg::utils::row_chunk_for_byte_budget(n, k + d),
            )?,
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
            penalty_range,
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
        gam_problem::validate_log_strengths(rho.iter().copied())
            .map_err(|error| invalid(format!("shared-tangent rho: {error}")))?;
        match &self.statistics {
            SufficientStatistics::Isotropic {
                root,
                projected_response,
            } => self.evaluate_isotropic(rho, root, projected_response),
            SufficientStatistics::Fisher { gram, cross } => self.evaluate_fisher(rho, gram, cross),
        }
    }

    fn evaluate_isotropic(
        &self,
        rho: &Array1<f64>,
        data_root: &Array2<f64>,
        projected_response: &Array2<f64>,
    ) -> Result<Evaluation, EstimationError> {
        let d = self.n_outputs;
        let (penalty, lambdas) = self.combined_penalty(rho)?;
        let spectrum = self.combined_penalty_spectrum(&penalty, &lambdas)?;
        let roots: Vec<Array2<f64>> = self
            .penalties
            .iter()
            .enumerate()
            .map(|(index, block)| {
                scaled_penalty_root(block, lambdas[index], self.n_coefficients, 1)
            })
            .collect();
        let augmented_rows =
            data_root.nrows() + roots.iter().map(|root| root.ncols()).sum::<usize>();
        let mut augmented = Array2::zeros((augmented_rows, self.n_coefficients));
        augmented
            .slice_mut(s![..data_root.nrows(), ..])
            .assign(data_root);
        let mut offset = data_root.nrows();
        for root in &roots {
            augmented
                .slice_mut(s![offset..offset + root.ncols(), ..])
                .assign(&root.t());
            offset += root.ncols();
        }
        let (orthogonal, upper) = augmented
            .qr()
            .map_err(EstimationError::LinearSystemSolveFailed)?;
        let (factor, log_determinant) = TangentPrecisionFactor::from_upper(upper)?;
        let rhs = orthogonal
            .slice(s![..data_root.nrows(), ..])
            .t()
            .dot(projected_response);
        let coefficients = factor.backsolve(&rhs);
        let profiled_deviance = self.profiled_deviance(&coefficients)?;
        let residual_degrees_of_freedom = self.residual_degrees_of_freedom(spectrum.rank)?;
        validate_profiled_deviance(profiled_deviance)?;

        let m = self.penalties.len();
        let (base_traces, base_cross_traces) = penalty_root_traces(&factor, &roots);
        let penalty_traces = base_traces * d as f64;
        let penalty_logdet_traces = &spectrum.traces * d as f64;
        let mut deviance_first = Array1::<f64>::zeros(m);
        let mut penalty_beta = Vec::with_capacity(m);
        for (index, root) in roots.iter().enumerate() {
            let root_beta = root.t().dot(&coefficients);
            deviance_first[index] = sum_products(&root_beta, &root_beta);
            penalty_beta.push(root.dot(&root_beta));
        }

        // REML profiles the scale out as `φ̂ = D_p/rdf` with `D_p` the PENALIZED
        // deviance, so the criterion's data term is `rdf·ln(D_p)` and its
        // ρ-derivative is `rdf·(β̂ᵗλⱼSⱼβ̂)/D_p` — exactly `deviance_first[j]/D_p`.
        // That identity is the ENVELOPE theorem, and it holds for `D_p` only:
        // `D_p` is stationary in β at β̂, the unpenalized `D` is not
        // (`dD/dρⱼ = 2β̂ᵗS_λA⁻¹λⱼSⱼβ̂`, a different quantity).
        //
        // The cost below used the UNPENALIZED `profiled_deviance` while the
        // gradient and Hessian were already the derivatives of the penalized one,
        // so the value and its derivatives described two different criteria. That
        // is why #2597's FD check missed by a large RATIO rather than by an
        // FD-step artifact: analytic `3.6460865888809835` against central FD
        // `0.6001259341301612` at `ρ = [−0.2, 0.35]`.
        //
        // `Σⱼ deviance_first[j] = β̂ᵗS_λβ̂` exactly, because `S_λ = Σⱼ λⱼSⱼ`, so the
        // penalized deviance needs no additional quadratic form.
        //
        // NOT changed here: the `sigma2 = evaluation.profiled_deviance/residual_df`
        // at the fit boundary is the same `D` vs `D_p` confusion in the SCALE
        // estimate, and REML's is `D_p/rdf`. It is left alone deliberately —
        // correcting it moves every reported standard error on this path, which
        // wants its own measurement rather than riding on an FD fixture.
        let penalized_deviance = profiled_deviance + deviance_first.sum();
        validate_profiled_deviance(penalized_deviance)?;

        let mut gradient = Array1::<f64>::zeros(m);
        for j in 0..m {
            gradient[j] = 0.5
                * (penalty_traces[j] - penalty_logdet_traces[j]
                    + residual_degrees_of_freedom * deviance_first[j] / penalized_deviance);
        }
        let mut hessian = Array2::<f64>::zeros((m, m));
        for j in 0..m {
            for kk in 0..=j {
                let h_cross = d as f64 * base_cross_traces[[j, kk]];
                let p_cross = d as f64 * spectrum.cross_traces[[j, kk]];
                let solved_penalty_beta = factor.solve_mat(&penalty_beta[kk]);
                let deviance_cross = sum_products(&penalty_beta[j], &solved_penalty_beta);
                let delta = usize::from(j == kk) as f64;
                let logdet_second = delta * penalty_traces[j] - h_cross;
                let penalty_logdet_second = delta * penalty_logdet_traces[j] - p_cross;
                let deviance_second = delta * deviance_first[j] - 2.0 * deviance_cross;
                let value = 0.5
                    * (logdet_second - penalty_logdet_second
                        + residual_degrees_of_freedom
                            * (deviance_second / penalized_deviance
                                - deviance_first[j] * deviance_first[kk]
                                    / (penalized_deviance * penalized_deviance)));
                hessian[[j, kk]] = value;
                hessian[[kk, j]] = value;
            }
        }
        let cost = 0.5
            * (d as f64 * log_determinant - d as f64 * spectrum.log_pseudo_determinant
                + residual_degrees_of_freedom
                    * (1.0
                        + (2.0 * std::f64::consts::PI * penalized_deviance
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
            lambdas,
            combined_penalty_rank: spectrum.rank,
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
        let spectrum = self.combined_penalty_spectrum(&penalty, &lambdas)?;
        let mut penalized = gram.clone();
        add_base_penalty_to_joint(&mut penalized, &penalty, d);
        let (factor, log_determinant) = spd_factor_and_logdet(&penalized)?;
        let beta = factor.solvevec(cross);
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
        let roots: Vec<Array2<f64>> = self
            .penalties
            .iter()
            .enumerate()
            .map(|(index, block)| scaled_penalty_root(block, lambdas[index], k, d))
            .collect();
        let (penalty_traces, base_cross_traces) = penalty_root_traces(&factor, &roots);
        let penalty_logdet_traces = &spectrum.traces * d as f64;
        let mut deviance_first = Array1::<f64>::zeros(m);
        let mut penalty_beta = Vec::with_capacity(m);
        for (index, root) in roots.iter().enumerate() {
            let root_beta = root.t().dot(&beta);
            deviance_first[index] = root_beta.dot(&root_beta);
            penalty_beta.push(root.dot(&root_beta));
        }

        // REML profiles the scale out as `φ̂ = D_p/rdf` with `D_p` the PENALIZED
        // deviance, so the criterion's data term is `rdf·ln(D_p)` and its
        // ρ-derivative is `rdf·(β̂ᵗλⱼSⱼβ̂)/D_p` — exactly `deviance_first[j]/D_p`.
        // That identity is the ENVELOPE theorem, and it holds for `D_p` only:
        // `D_p` is stationary in β at β̂, the unpenalized `D` is not
        // (`dD/dρⱼ = 2β̂ᵗS_λA⁻¹λⱼSⱼβ̂`, a different quantity).
        //
        // The cost below used the UNPENALIZED `profiled_deviance` while the
        // gradient and Hessian were already the derivatives of the penalized one,
        // so the value and its derivatives described two different criteria. That
        // is why #2597's FD check missed by a large RATIO rather than by an
        // FD-step artifact: analytic `3.6460865888809835` against central FD
        // `0.6001259341301612` at `ρ = [−0.2, 0.35]`.
        //
        // `Σⱼ deviance_first[j] = β̂ᵗS_λβ̂` exactly, because `S_λ = Σⱼ λⱼSⱼ`, so the
        // penalized deviance needs no additional quadratic form.
        //
        // NOT changed here: the `sigma2 = evaluation.profiled_deviance/residual_df`
        // at the fit boundary is the same `D` vs `D_p` confusion in the SCALE
        // estimate, and REML's is `D_p/rdf`. It is left alone deliberately —
        // correcting it moves every reported standard error on this path, which
        // wants its own measurement rather than riding on an FD fixture.
        let penalized_deviance = profiled_deviance + deviance_first.sum();
        validate_profiled_deviance(penalized_deviance)?;

        let mut gradient = Array1::<f64>::zeros(m);
        for j in 0..m {
            gradient[j] = 0.5
                * (penalty_traces[j] - penalty_logdet_traces[j]
                    + residual_degrees_of_freedom * deviance_first[j] / penalized_deviance);
        }
        let mut hessian = Array2::<f64>::zeros((m, m));
        for j in 0..m {
            for kk in 0..=j {
                let h_cross = base_cross_traces[[j, kk]];
                let p_cross = d as f64 * spectrum.cross_traces[[j, kk]];
                let solved_penalty_beta = factor.solvevec(&penalty_beta[kk]);
                let deviance_cross = penalty_beta[j].dot(&solved_penalty_beta);
                let delta = usize::from(j == kk) as f64;
                let logdet_second = delta * penalty_traces[j] - h_cross;
                let penalty_logdet_second = delta * penalty_logdet_traces[j] - p_cross;
                let deviance_second = delta * deviance_first[j] - 2.0 * deviance_cross;
                let value = 0.5
                    * (logdet_second - penalty_logdet_second
                        + residual_degrees_of_freedom
                            * (deviance_second / penalized_deviance
                                - deviance_first[j] * deviance_first[kk]
                                    / (penalized_deviance * penalized_deviance)));
                hessian[[j, kk]] = value;
                hessian[[kk, j]] = value;
            }
        }
        let cost = 0.5
            * (log_determinant - d as f64 * spectrum.log_pseudo_determinant
                + residual_degrees_of_freedom
                    * (1.0
                        + (2.0 * std::f64::consts::PI * penalized_deviance
                            / residual_degrees_of_freedom)
                            .ln()));
        if beta.len() != q {
            return Err(invalid("internal Fisher solution shape mismatch"));
        }
        validate_evaluation(cost, &gradient, &hessian)?;
        Ok(Evaluation {
            cost,
            gradient,
            hessian,
            coefficients,
            profiled_deviance,
            penalty_traces,
            lambdas,
            combined_penalty_rank: spectrum.rank,
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

    fn combined_penalty_spectrum(
        &self,
        penalty: &Array2<f64>,
        lambdas: &Array1<f64>,
    ) -> Result<PenaltySpectrum, EstimationError> {
        let rank = self.penalty_range.ncols();
        if rank == 0 {
            return Ok(PenaltySpectrum {
                rank,
                log_pseudo_determinant: 0.0,
                traces: Array1::zeros(self.penalties.len()),
                cross_traces: Array2::zeros((self.penalties.len(), self.penalties.len())),
            });
        }
        // The restriction is positive definite for every positive lambda.
        // Reclassifying its eigenvalues against the largest scaled curvature
        // would change REML's nullity and objective as one strength shrinks.
        let restricted = self
            .penalty_range
            .t()
            .dot(&penalty.dot(&self.penalty_range));
        let (factor, log_pseudo_determinant) = spd_factor_and_logdet(&restricted)?;
        let roots: Vec<Array2<f64>> = self
            .penalties
            .iter()
            .enumerate()
            .map(|(index, block)| {
                let root = scaled_penalty_root(block, lambdas[index], self.n_coefficients, 1);
                self.penalty_range.t().dot(&root)
            })
            .collect();
        let (traces, cross_traces) = penalty_root_traces(&factor, &roots);
        Ok(PenaltySpectrum {
            rank,
            log_pseudo_determinant,
            traces,
            cross_traces,
        })
    }

    fn combined_penalty(
        &self,
        rho: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>), EstimationError> {
        let mut combined = Array2::<f64>::zeros((self.n_coefficients, self.n_coefficients));
        let lambdas = Array1::from_vec(
            gam_problem::checked_exp_log_strengths(rho.iter().copied())
                .map_err(|error| invalid(format!("shared-tangent rho: {error}")))?,
        );
        for (index, penalty) in self.penalties.iter().enumerate() {
            let lambda = lambdas[index];
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
        let analysis = gam_terms::basis::analyze_penalty_block(&penalty.matrix)
            .map_err(|error| invalid(format!("shared-tangent penalty {slot}: {error}")))?;
        if analysis.negative_dim != 0 {
            return Err(invalid(format!(
                "shared-tangent penalty {slot} must be positive semidefinite"
            )));
        }
        if analysis.rank == 0 {
            continue;
        }
        let active: Vec<usize> = analysis
            .eigenvalues
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value > analysis.rank_tol).then_some(index))
            .collect();
        let root = Array2::from_shape_fn((active.len(), q), |(mode, col)| {
            let index = active[mode];
            analysis.eigenvalues[index].sqrt() * analysis.eigenvectors[[col, index]]
        });
        prepared.push(PreparedPenalty {
            output_slot: slot,
            column_start: penalty.column_start,
            rank: root.nrows(),
            local: root.t().dot(&root),
            root,
        });
    }
    Ok(prepared)
}

fn assemble_isotropic_statistics(
    design: &DesignMatrix,
    response: &Array2<f64>,
    weights: &Array1<f64>,
    chunk_rows: usize,
) -> Result<SufficientStatistics, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let d = response.ncols();
    let mut root = Array2::<f64>::zeros((0, k));
    let mut projected_response = Array2::<f64>::zeros((0, d));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let x_chunk = design
            .try_row_chunk(start..end)
            .map_err(|error| invalid(format!("failed to read design row chunk: {error}")))?;
        validate_design_chunk(&x_chunk)?;
        let live_rows: Vec<usize> = (start..end).filter(|&row| weights[row] > 0.0).collect();
        if live_rows.is_empty() {
            continue;
        }
        let retained_rows = root.nrows();
        let mut stacked = Array2::zeros((retained_rows + live_rows.len(), k));
        let mut rhs = Array2::zeros((stacked.nrows(), d));
        stacked.slice_mut(s![..retained_rows, ..]).assign(&root);
        rhs.slice_mut(s![..retained_rows, ..])
            .assign(&projected_response);
        for (local, &row) in live_rows.iter().enumerate() {
            let scale = weights[row].sqrt();
            for col in 0..k {
                stacked[[retained_rows + local, col]] = scale * x_chunk[[row - start, col]];
            }
            for output in 0..d {
                rhs[[retained_rows + local, output]] = scale * response[[row, output]];
            }
        }
        let (orthogonal, upper) = stacked
            .qr()
            .map_err(EstimationError::LinearSystemSolveFailed)?;
        projected_response = orthogonal.t().dot(&rhs);
        root = upper;
    }
    Ok(SufficientStatistics::Isotropic {
        root,
        projected_response,
    })
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

struct TangentPrecisionFactor {
    /// Precision is RᵀR. Keeping R avoids squaring the design condition number.
    upper: Array2<f64>,
}

impl TangentPrecisionFactor {
    fn from_upper(upper: Array2<f64>) -> Result<(Self, f64), EstimationError> {
        if upper.nrows() != upper.ncols()
            || upper.iter().any(|value| !value.is_finite())
            || upper.diag().iter().any(|value| *value == 0.0)
        {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        let log_determinant = 2.0
            * upper
                .diag()
                .iter()
                .map(|value| value.abs().ln())
                .sum::<f64>();
        if !log_determinant.is_finite() {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        Ok((Self { upper }, log_determinant))
    }

    fn whiten(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut out = rhs.clone();
        FaerArrayView::new(&self.upper)
            .as_ref()
            .transpose()
            .solve_lower_triangular_in_place(array2_to_matmut(&mut out));
        out
    }

    fn backsolve(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut out = rhs.clone();
        FaerArrayView::new(&self.upper)
            .as_ref()
            .solve_upper_triangular_in_place(array2_to_matmut(&mut out));
        out
    }

    fn solve_mat(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.backsolve(&self.whiten(rhs))
    }

    fn solvevec(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.solve_mat(&rhs.clone().insert_axis(ndarray::Axis(1)))
            .column(0)
            .to_owned()
    }
}

fn spd_factor_and_logdet(
    matrix: &Array2<f64>,
) -> Result<(TangentPrecisionFactor, f64), EstimationError> {
    let factor =
        matrix
            .cholesky(Side::Lower)
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    TangentPrecisionFactor::from_upper(factor.lower_triangular().t().to_owned())
}

/// Columns are the scaled penalty root, with the output identity implicit
/// on the isotropic path and lifted only for varying Fisher geometry.
fn scaled_penalty_root(
    penalty: &PreparedPenalty,
    lambda: f64,
    n_coefficients: usize,
    n_outputs: usize,
) -> Array2<f64> {
    let mut root = Array2::zeros((n_coefficients * n_outputs, penalty.rank * n_outputs));
    let scale = lambda.sqrt();
    for mode in 0..penalty.rank {
        for col in 0..penalty.root.ncols() {
            for output in 0..n_outputs {
                root[[
                    (penalty.column_start + col) * n_outputs + output,
                    mode * n_outputs + output,
                ]] = scale * penalty.root[[mode, col]];
            }
        }
    }
    root
}

/// tr(A⁻¹ Rj Rjᵀ) and tr(A⁻¹ Rj Rjᵀ A⁻¹ Rk Rkᵀ).
/// The cross trace is ||Rjᵀ A⁻¹ Rk||²_F, so no dense inverse
/// sandwiches or cancellation between their large entries is required.
fn penalty_root_traces(
    factor: &TangentPrecisionFactor,
    roots: &[Array2<f64>],
) -> (Array1<f64>, Array2<f64>) {
    let whitened: Vec<Array2<f64>> = roots.iter().map(|root| factor.whiten(root)).collect();
    let mut traces = Array1::zeros(roots.len());
    let mut cross_traces = Array2::zeros((roots.len(), roots.len()));
    for j in 0..roots.len() {
        traces[j] = sum_products(&whitened[j], &whitened[j]);
        for k in 0..=j {
            let product = whitened[j].t().dot(&whitened[k]);
            let value = sum_products(&product, &product);
            cross_traces[[j, k]] = value;
            cross_traces[[k, j]] = value;
        }
    }
    (traces, cross_traces)
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
#[path = "response_geometry_rotation_tests.rs"]
mod rotation_tests;

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
