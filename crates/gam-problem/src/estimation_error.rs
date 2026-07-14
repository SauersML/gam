use gam_linalg::LinalgError;
use gam_linalg::faer_ndarray::FaerLinalgError;
use serde::{Deserialize, Serialize};

use crate::{BasisError, CustomFamilyError, MonotoneRootError};

/// Fixed-lambda solver stage that owns a resumable coefficient checkpoint.
///
/// The multinomial fitter has two distinct objectives: the ordinary softmax
/// likelihood and the Firth/Jeffreys separation refit. Recording the stage is
/// therefore part of correctness: a Firth checkpoint must resume the Firth
/// objective rather than being mistaken for an ordinary multinomial start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedLambdaSolverStage {
    BinomialMultiNewton,
    MultinomialNewton,
    MultinomialFirth,
}

impl core::fmt::Display for FixedLambdaSolverStage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::BinomialMultiNewton => "binomial-multi Newton",
            Self::MultinomialNewton => "multinomial Newton",
            Self::MultinomialFirth => "multinomial Firth/Jeffreys Newton",
        })
    }
}

/// Exhaustive terminal reason for a fixed-lambda solve without a certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedLambdaStallReason {
    IterationBudgetExhausted,
    LineSearchExhausted,
    StationarityCertificateFailed,
}

impl core::fmt::Display for FixedLambdaStallReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::IterationBudgetExhausted => "iteration budget exhausted",
            Self::LineSearchExhausted => "line search exhausted without an accepted step",
            Self::StationarityCertificateFailed => "stationarity certificate failed",
        })
    }
}

/// Solver-native first-order residual carried by a fixed-lambda stall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedLambdaResidualKind {
    /// Euclidean norm of the exact penalized likelihood gradient.
    PenalizedGradientNorm,
    /// Firth/Jeffreys Newton decrement `0.5 * |score' H^-1 score|`.
    NewtonDecrement,
}

impl core::fmt::Display for FixedLambdaResidualKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::PenalizedGradientNorm => "penalized gradient norm",
            Self::NewtonDecrement => "Newton decrement",
        })
    }
}

/// Evidence from the exact stationarity check at the last accepted iterate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FixedLambdaStationarityEvidence {
    pub kind: FixedLambdaResidualKind,
    pub residual: f64,
    pub bound: f64,
}

impl core::fmt::Display for FixedLambdaStationarityEvidence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {:.6e} against bound {:.6e}",
            self.kind, self.residual, self.bound
        )
    }
}

/// Owned, serde-safe coefficient checkpoint for a fixed-lambda Newton solve.
///
/// Coefficients are row-major with shape `(rows, cols)`, where rows are the
/// per-output coefficient count and columns are active outputs/classes. The
/// values deliberately remain private so diagnostics cannot accidentally dump
/// a potentially large coefficient vector; resume code accesses them through
/// [`Self::values`] after calling [`Self::validate`].
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedLambdaCheckpoint {
    stage: FixedLambdaSolverStage,
    coefficients_row_major: Vec<f64>,
    rows: usize,
    cols: usize,
    completed_iterations: usize,
}

impl FixedLambdaCheckpoint {
    pub fn new(
        stage: FixedLambdaSolverStage,
        coefficients_row_major: Vec<f64>,
        rows: usize,
        cols: usize,
        completed_iterations: usize,
    ) -> Result<Self, String> {
        let checkpoint = Self {
            stage,
            coefficients_row_major,
            rows,
            cols,
            completed_iterations,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Validate persisted checkpoint geometry and coefficient finiteness before
    /// rebuilding an ndarray view in a resumed solver.
    pub fn validate(&self) -> Result<(), String> {
        if self.rows == 0 || self.cols == 0 {
            return Err(format!(
                "fixed-lambda checkpoint shape must be nonempty, got {}x{}",
                self.rows, self.cols
            ));
        }
        let expected = self.rows.checked_mul(self.cols).ok_or_else(|| {
            format!(
                "fixed-lambda checkpoint shape {}x{} overflows usize",
                self.rows, self.cols
            )
        })?;
        if self.coefficients_row_major.len() != expected {
            return Err(format!(
                "fixed-lambda checkpoint has {} coefficient values, expected {} for shape {}x{}",
                self.coefficients_row_major.len(),
                expected,
                self.rows,
                self.cols
            ));
        }
        if let Some((index, _)) = self
            .coefficients_row_major
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "fixed-lambda checkpoint coefficient {index} must be finite"
            ));
        }
        Ok(())
    }

    pub fn stage(&self) -> FixedLambdaSolverStage {
        self.stage
    }

    pub fn values(&self) -> &[f64] {
        &self.coefficients_row_major
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn completed_iterations(&self) -> usize {
        self.completed_iterations
    }
}

impl core::fmt::Display for FixedLambdaCheckpoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} checkpoint {}x{} after {} iteration(s)",
            self.stage, self.rows, self.cols, self.completed_iterations
        )
    }
}

impl core::fmt::Debug for FixedLambdaCheckpoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

/// A comprehensive error type for the model estimation process.
#[derive(thiserror::Error)]
pub enum EstimationError {
    #[error(transparent)]
    InvalidStabilization(#[from] crate::InvalidStabilization),

    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] BasisError),

    #[error("Custom-family fit failed: {0}")]
    CustomFamily(#[from] CustomFamilyError),

    #[error("A linear system solve failed. The penalized Hessian may be singular. Error: {0}")]
    LinearSystemSolveFailed(FaerLinalgError),

    #[error("Eigendecomposition failed: {0}")]
    EigendecompositionFailed(FaerLinalgError),

    #[error(
        "Penalty spectrum check failed in '{context}': non-finite eigenvalue {value:?} at index {index}"
    )]
    PenaltySpectrumNonFinite {
        context: String,
        index: usize,
        value: f64,
    },

    #[error(
        "Penalty spectrum check failed in '{context}': indefinite eigenvalue {value:.3e} at index {index} (tolerance {tolerance:.3e}, scale {scale:.3e})"
    )]
    PenaltySpectrumIndefinite {
        context: String,
        index: usize,
        value: f64,
        tolerance: f64,
        scale: f64,
    },

    #[error("Parameter constraint violation: {0}")]
    ParameterConstraintViolation(String),

    #[error(
        "The P-IRLS inner loop did not converge within {max_iterations} iterations. Last gradient norm was {last_change:.6e}."
    )]
    PirlsDidNotConverge {
        max_iterations: usize,
        last_change: f64,
    },

    #[error(
        "{context} did not certify a stationary fixed-lambda optimum after {} iteration(s): \
         {reason}; final minimized objective {objective_value:.6e}; {stationarity}. A fit is \
         only minted from a converged optimization; resume by passing the carried checkpoint \
         through the fixed-lambda input's `resume_from` field ({checkpoint}).",
        .checkpoint.completed_iterations()
    )]
    FixedLambdaNewtonDidNotConverge {
        /// Which fixed-λ Newton entry stalled (e.g. the multinomial softmax or
        /// independent-binomial vector-GLM solve, or the Firth refit lane).
        context: String,
        /// Why the solver stopped without its convergence certificate.
        reason: FixedLambdaStallReason,
        /// Final value of the solver's minimized criterion. For ordinary vector
        /// GLMs this is `-log L + penalty`; for the Firth lane it also includes
        /// the negative Jeffreys `0.5 log det(I)` contribution.
        objective_value: f64,
        /// Exact first-order residual and the bound it failed to clear.
        stationarity: FixedLambdaStationarityEvidence,
        /// Last accepted coefficients and cumulative iteration count. This is
        /// work-preservation state, not a fitted model, and carries no covariance
        /// or prediction surface.
        checkpoint: FixedLambdaCheckpoint,
    },

    #[error(
        "Block-orthogonal Gaussian REML did not converge within {iterations} outer passes: \
         max relative rho-score residual {max_score_residual:.6e}/{score_tol:.3e}, \
         minimum profiled curvature {min_profile_curvature:.6e} (negative allowance \
         {profile_curvature_roundoff:.3e}; last scale fixed-point step \
         {last_scale_step:.6e}{}). \
         A fit is only minted from a converged optimization; resume from the \
         checkpoint by passing `init_rhos` = {rho_checkpoint:?}.",
        if *cycle_detected { ", deterministic limit cycle detected" } else { "" }
    )]
    BlockOrthogonalRemlDidNotConverge {
        /// Outer alternation passes executed before exhaustion.
        iterations: usize,
        /// Largest per-block |dV/drho| at the final iterate, normalized by the
        /// score's natural magnitude `d * max(1, rank)`.
        max_score_residual: f64,
        /// Tolerance the residual had to meet for the convergence certificate.
        score_tol: f64,
        /// Smallest eigenvalue of the analytic rho Hessian after profiling out
        /// the exact conditional scale block.
        min_profile_curvature: f64,
        /// Dimension-scaled eigensolver roundoff allowed below zero when
        /// certifying positive semidefiniteness.
        profile_curvature_roundoff: f64,
        /// Last max |Δ log scale-precision| fixed-point movement (evidence of
        /// whether the alternation was still moving or had stalled).
        last_scale_step: f64,
        /// The alternation revisited an earlier `(rho, scale)` state exactly;
        /// as a deterministic map it can never certify, so it stopped early.
        cycle_detected: bool,
        /// Per-block log-lambda iterates at exhaustion; feed back through the
        /// entry point's `init_rhos` to resume rather than restart.
        rho_checkpoint: Vec<f64>,
    },

    #[error(
        "Negative-binomial (theta, rho) optimization did not certify a joint optimum within \
         {rounds} round(s): projected rho-gradient {rho_projected_grad_norm:.3e} against \
         {rho_stationarity_bound:.3e}, theta-score Newton residual {theta_score_residual:.3e} \
         against {theta_stationarity_bound:.3e}. A fit is only minted when both analytic \
         partials are stationary at one identical point; resume from theta={theta_checkpoint:.6e} \
         and rho={rho_checkpoint:?}."
    )]
    NegativeBinomialAlternationDidNotConverge {
        /// Joint block-coordinate rounds executed before exhaustion.
        rounds: usize,
        /// Conditional theta coordinate at the best measured checkpoint.
        theta_checkpoint: f64,
        /// KKT-projected rho-gradient norm at that checkpoint.
        rho_projected_grad_norm: f64,
        /// Bound the rho residual had to clear.
        rho_stationarity_bound: f64,
        /// Curvature-normalized log-theta score residual at that checkpoint.
        theta_score_residual: f64,
        /// Bound the theta residual had to clear.
        theta_stationarity_bound: f64,
        /// Best measured log-smoothing checkpoint for warm-started resume.
        rho_checkpoint: Vec<f64>,
    },

    #[error(
        "Perfect or quasi-perfect separation detected during model fitting at iteration {iteration}. \
        The model cannot converge because a predictor perfectly separates the binary outcomes. \
        (Diagnostic: max|eta| = {max_abs_eta:.2e})."
    )]
    PerfectSeparationDetected { iteration: usize, max_abs_eta: f64 },

    #[error(
        "Pre-fit perfect separation detected in the realized binomial inverse-link design: column {column_index} \
        has a threshold {threshold:.6e} that separates the binary outcomes \
        (positive_above_threshold={positive_above_threshold}). The unpenalized MLE is not finite; \
        enable Firth/Jeffreys bias reduction or remove/reparameterize the separating column."
    )]
    PrefitPerfectSeparationDetected {
        column_index: usize,
        threshold: f64,
        positive_above_threshold: bool,
    },

    #[error(
        "Pre-fit linear separation detected in the realized binomial inverse-link design: \
        {num_unpenalized_columns} effectively unpenalized columns admit a separating direction \
        with minimum signed margin {min_signed_margin:.6e} (columns {column_indices:?}). \
        The unpenalized MLE is not finite; enable Firth/Jeffreys bias reduction or \
        remove/reparameterize/penalize the separating columns."
    )]
    PrefitLinearSeparationDetected {
        min_signed_margin: f64,
        num_unpenalized_columns: usize,
        column_indices: Vec<usize>,
    },

    #[error(
        "Pre-fit rank deficiency detected in the realized unpenalized design: rank {rank} < {num_unpenalized_columns} \
        unpenalized columns (min eigenvalue {min_eigenvalue:.3e}, tolerance {tolerance:.3e}, columns {column_indices:?}). \
        Remove/reparameterize the aliased columns or add an explicit penalty/constraint before fitting."
    )]
    PrefitRankDeficientDesignDetected {
        rank: usize,
        num_unpenalized_columns: usize,
        min_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },

    #[error(
        "Pre-fit near-degeneracy detected in the realized unpenalized design: the {num_unpenalized_columns} \
        unpenalized columns span a numerically rank-degenerate direction (Gram condition number {condition_number:.3e} \
        exceeds tolerance {tolerance:.3e}; min eigenvalue {min_eigenvalue:.3e}, max eigenvalue {max_eigenvalue:.3e}, \
        columns {column_indices:?}). The unpenalized normal equations are effectively singular along this direction, \
        so the fit would grind/diverge. Remove/reparameterize the near-aliased columns or add an explicit \
        penalty/constraint before fitting."
    )]
    PrefitNearDegenerateDesignDetected {
        num_unpenalized_columns: usize,
        condition_number: f64,
        min_eigenvalue: f64,
        max_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },

    #[error(
        "Perfect or quasi-perfect separation detected during multinomial fitting at iteration {iteration}. \
        The active class-{active_class_index} logit against the reference class is saturated at training row {row_index}, \
        so the unpenalized softmax MLE is not finite in that direction. \
        (Diagnostic: max|eta| = {max_abs_eta:.2e})."
    )]
    MultinomialSeparationDetected {
        iteration: usize,
        max_abs_eta: f64,
        active_class_index: usize,
        row_index: usize,
    },

    #[error(
        "Hessian matrix is not positive definite (minimum eigenvalue: {min_eigenvalue:.4e}). This indicates a numerical instability."
    )]
    HessianNotPositiveDefinite { min_eigenvalue: f64 },

    #[error("REML smoothing optimization failed to converge: {0}")]
    RemlOptimizationFailed(String),

    #[error("Fatal outer-objective evaluation failure ({context}): {source}")]
    OuterObjectiveEvaluationFailed {
        context: String,
        #[source]
        source: Box<EstimationError>,
    },

    #[error(
        "Outer smoothing-parameter optimization did not certify a stationary optimum \
         ({context}): {reason} after {iterations} outer iteration(s); final objective \
         {final_value:.6e}, projected gradient norm {} against stationarity bound \
         {stationarity_bound:.3e}. A fit is only minted from a converged optimization; \
         the best iterate is carried as a checkpoint — resume by seeding the outer \
         search at rho_checkpoint = {rho_checkpoint:?}.",
        .projected_grad_norm.map_or_else(|| "unmeasured".to_string(), |g| format!("{g:.3e}"))
    )]
    RemlDidNotConverge {
        /// Fit context label (the same string the outer runner logs under).
        context: String,
        /// Which certificate failed: budget exhaustion, line-search collapse,
        /// non-stationary cost stall, or a failed post-solve stationarity
        /// certificate.
        reason: String,
        /// Outer iterations executed across all solver restarts.
        iterations: usize,
        /// Objective value at the abandoned best iterate.
        final_value: f64,
        /// KKT-projected gradient norm at the best iterate, when the solver
        /// measured a gradient there (`None` for gradient-free exits).
        projected_grad_norm: Option<f64>,
        /// Bound the projected gradient had to clear for the stationarity
        /// certificate.
        stationarity_bound: f64,
        /// Best (lowest-objective feasible) outer iterate at exhaustion. This
        /// is work-preservation evidence for resume — it is NOT a fit and no
        /// fitted-model API is reachable from it.
        rho_checkpoint: Vec<f64>,
    },

    #[error(
        "Fit assembly rejected a non-converged optimization state: inner status \
         {inner_status}, outer status {outer_status}, after {outer_iterations} outer \
         iteration(s); final objective {final_value:.6e}; stationarity residual \
         {stationarity_residual:?} against {stationarity_bound:?}, step residual \
         {step_residual:?} against {step_bound:?}. The best rho checkpoint is \
         {rho_checkpoint:?} and the resume token is {resume_token:?}; no fitted-model \
         API was constructed."
    )]
    FitDidNotConverge {
        /// Diagnostic inner-solver terminal status. This is deliberately a
        /// string at the neutral problem layer; concrete solver status enums
        /// live in downstream fitting crates.
        inner_status: String,
        /// Outer terminal/certificate verdict.
        outer_status: String,
        /// Completed outer iterations at the rejected checkpoint.
        outer_iterations: usize,
        /// Objective value at the best available checkpoint.
        final_value: f64,
        /// Exact analytic first-order gradient or root-equivalent fixed-point
        /// residual, when it was measured.
        stationarity_residual: Option<f64>,
        /// Bound the first-order residual had to clear.
        stationarity_bound: Option<f64>,
        /// Final accepted-step residual, when the solver exported it.
        step_residual: Option<f64>,
        /// Bound the step residual had to clear.
        step_bound: Option<f64>,
        /// Work-preserving smoothing checkpoint; this is not a fit.
        rho_checkpoint: Vec<f64>,
        /// Opaque durable-cache resume token, when checkpoint persistence was
        /// enabled for the failed run.
        resume_token: Option<String>,
    },

    #[error("{context}: unified evaluator returned no gradient in {mode} mode")]
    GradientUnavailable {
        context: &'static str,
        mode: &'static str,
    },

    #[error("An internal error occurred during model layout or coefficient mapping: {0}")]
    LayoutError(String),

    #[error(
        "Model is over-parameterized: {num_coeffs} coefficients for {num_samples} samples.\n\n\
        Coefficient Breakdown:\n\
          - Intercept:                     {intercept_coeffs}\n\
          - Binary Main Effects:           {binary_main_coeffs}\n\
          - Primary Smooth Effects:        {primary_smooth_coeffs}\n\
          - Binary×Primary Interactions:   {binary_primary_interaction_coeffs}\n\
          - Auxiliary Main Effects:        {aux_main_coeffs}\n\
          - Auxiliary Interactions:        {aux_interaction_coeffs}"
    )]
    ModelOverparameterized {
        num_coeffs: usize,
        num_samples: usize,
        intercept_coeffs: usize,
        binary_main_coeffs: usize,
        primary_smooth_coeffs: usize,
        aux_main_coeffs: usize,
        binary_primary_interaction_coeffs: usize,
        aux_interaction_coeffs: usize,
    },

    #[error(
        "Model is ill-conditioned with condition number {condition_number:.2e}. This typically occurs when the model is over-parameterized (too many knots relative to data points). Consider reducing the number of knots or increasing regularization."
    )]
    ModelIsIllConditioned { condition_number: f64 },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error(
        "Inverse-link domain violation for {link}: eta={eta:?} is outside the supported \
         interval [{lower}, {upper}]"
    )]
    InverseLinkDomainViolation {
        link: &'static str,
        eta: f64,
        lower: f64,
        upper: f64,
    },

    #[error(
        "PIRLS row geometry is not representable at row {row}: {quantity} evaluated from \
         eta={eta:?} produced {value:?}"
    )]
    PirlsRowGeometryUnrepresentable {
        row: usize,
        quantity: &'static str,
        eta: f64,
        value: f64,
    },

    #[error(
        "Exact Tweedie series work limit at row {row}: at least {required_terms_lower_bound:?} terms are required, budget is {budget}"
    )]
    ExactTweedieSeriesWorkLimit {
        row: usize,
        required_terms_lower_bound: f64,
        budget: usize,
    },

    #[error(
        "Log-strength domain violation at coordinate {coordinate}: value={value:?} is outside \
         the supported interval [{lower}, {upper}]"
    )]
    LogStrengthDomainViolation {
        coordinate: usize,
        value: f64,
        lower: f64,
        upper: f64,
    },

    #[error("monotone root solve: {0}")]
    MonotoneRoot(#[from] MonotoneRootError),

    #[error("Calibrator training failed: {0}")]
    CalibratorTrainingFailed(String),

    #[error("Invalid specification: {0}")]
    InvalidSpecification(String),

    #[error("Prediction error")]
    PredictionError,
}

// Ensure Debug prints with actual line breaks by delegating to Display
impl core::fmt::Debug for EstimationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self)
    }
}

impl EstimationError {
    /// Preserve a thrown outer-objective failure across seed, solver, and
    /// fallback-plan orchestration. Trial-domain refusals must be represented
    /// as a finite API outcome (`+inf` / `OuterEval::infeasible`); an `Err`
    /// means the evaluation artifact itself could not be constructed and must
    /// never be retried as another numerical point.
    pub fn fatal_outer_evaluation(
        context: impl Into<String>,
        source: EstimationError,
    ) -> Self {
        if matches!(
            &source,
            EstimationError::OuterObjectiveEvaluationFailed { .. }
        ) {
            source
        } else {
            EstimationError::OuterObjectiveEvaluationFailed {
                context: context.into(),
                source: Box::new(source),
            }
        }
    }

    pub fn is_fatal_outer_evaluation(&self) -> bool {
        matches!(self, EstimationError::OuterObjectiveEvaluationFailed { .. })
    }

    /// Classifies inner-solve failures that the outer REML loop should
    /// treat as a soft retreat (return +inf cost / infeasible outer-eval)
    /// rather than propagate as a hard error.
    ///
    /// Why: when the penalised Hessian becomes effectively singular at the
    /// current rho, when P-IRLS hits a perfect-separation diagnostic, or when
    /// it exhausts its iteration budget, the outer optimiser's correct
    /// response is to back away from this rho — not to terminate the fit.
    /// All three variants encode "the inner problem at this rho is too hard
    /// to evaluate, try a different rho".
    pub fn is_inner_solve_retreat(&self) -> bool {
        matches!(
            self,
            EstimationError::ModelIsIllConditioned { .. }
                | EstimationError::PerfectSeparationDetected { .. }
                | EstimationError::MultinomialSeparationDetected { .. }
                | EstimationError::PirlsDidNotConverge { .. }
                | EstimationError::FixedLambdaNewtonDidNotConverge { .. }
        )
    }
}

impl From<LinalgError> for EstimationError {
    fn from(error: LinalgError) -> Self {
        match error {
            LinalgError::InvalidInput(message) => EstimationError::InvalidInput(message),
            LinalgError::HessianNotPositiveDefinite { min_eigenvalue } => {
                EstimationError::HessianNotPositiveDefinite { min_eigenvalue }
            }
            LinalgError::ModelIsIllConditioned { condition_number } => {
                EstimationError::ModelIsIllConditioned { condition_number }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_inner_solve_retreat ────────────────────────────────────────────────

    #[test]
    fn model_ill_conditioned_is_retreat() {
        assert!(
            EstimationError::ModelIsIllConditioned {
                condition_number: 1e15
            }
            .is_inner_solve_retreat()
        );
    }

    #[test]
    fn perfect_separation_is_retreat() {
        assert!(
            EstimationError::PerfectSeparationDetected {
                iteration: 3,
                max_abs_eta: 50.0
            }
            .is_inner_solve_retreat()
        );
    }

    #[test]
    fn multinomial_separation_is_retreat() {
        assert!(
            EstimationError::MultinomialSeparationDetected {
                iteration: 1,
                max_abs_eta: 100.0,
                active_class_index: 2,
                row_index: 7
            }
            .is_inner_solve_retreat()
        );
    }

    #[test]
    fn pirls_did_not_converge_is_retreat() {
        assert!(
            EstimationError::PirlsDidNotConverge {
                max_iterations: 100,
                last_change: 1e-3
            }
            .is_inner_solve_retreat()
        );
    }

    #[test]
    fn invalid_input_is_not_retreat() {
        assert!(!EstimationError::InvalidInput("bad".to_string()).is_inner_solve_retreat());
    }

    #[test]
    fn reml_optimization_failed_is_not_retreat() {
        assert!(
            !EstimationError::RemlOptimizationFailed("outer fail".to_string())
                .is_inner_solve_retreat()
        );
    }

    #[test]
    fn fatal_outer_evaluation_is_typed_and_idempotent() {
        let error = EstimationError::fatal_outer_evaluation(
            "seed screening",
            EstimationError::InvalidInput("frame mismatch".to_string()),
        );
        assert!(error.is_fatal_outer_evaluation());
        assert!(error.to_string().contains("frame mismatch"));

        let nested = EstimationError::fatal_outer_evaluation("fallback plan", error);
        assert!(nested.is_fatal_outer_evaluation());
        assert_eq!(
            nested.to_string().matches("Fatal outer-objective").count(),
            1,
            "fatal provenance must not be re-wrapped at every orchestration layer"
        );
    }

    // ── error message content ─────────────────────────────────────────────────

    #[test]
    fn invalid_input_message_appears_in_display() {
        let err = EstimationError::InvalidInput("test_message".to_string());
        assert!(err.to_string().contains("test_message"));
    }

    #[test]
    fn pirls_did_not_converge_mentions_max_iterations() {
        let err = EstimationError::PirlsDidNotConverge {
            max_iterations: 42,
            last_change: 0.001,
        };
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn fixed_lambda_checkpoint_validates_shape_and_values() {
        let checkpoint = FixedLambdaCheckpoint::new(
            FixedLambdaSolverStage::MultinomialNewton,
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            2,
            7,
        )
        .expect("well-shaped finite checkpoint");
        assert_eq!(
            checkpoint.stage(),
            FixedLambdaSolverStage::MultinomialNewton
        );
        assert_eq!(checkpoint.values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!((checkpoint.rows(), checkpoint.cols()), (2, 2));
        assert_eq!(checkpoint.completed_iterations(), 7);

        assert!(
            FixedLambdaCheckpoint::new(
                FixedLambdaSolverStage::BinomialMultiNewton,
                vec![1.0],
                2,
                1,
                0,
            )
            .is_err(),
            "coefficient length must match rows * cols"
        );
        assert!(
            FixedLambdaCheckpoint::new(
                FixedLambdaSolverStage::BinomialMultiNewton,
                vec![f64::NAN],
                1,
                1,
                0,
            )
            .is_err(),
            "checkpoint coefficients must be finite"
        );
        assert!(
            FixedLambdaCheckpoint::new(
                FixedLambdaSolverStage::MultinomialFirth,
                Vec::new(),
                usize::MAX,
                2,
                0,
            )
            .is_err(),
            "checkpoint shape multiplication must not overflow"
        );
    }

    #[test]
    fn fixed_lambda_error_displays_evidence_but_never_coefficients() {
        let checkpoint = FixedLambdaCheckpoint::new(
            FixedLambdaSolverStage::MultinomialFirth,
            vec![12_345.678_9, -98_765.432_1],
            2,
            1,
            11,
        )
        .expect("valid checkpoint");
        let checkpoint_debug = format!("{checkpoint:?}");
        assert!(!checkpoint_debug.contains("12345.6789"));
        assert!(!checkpoint_debug.contains("98765.4321"));
        let err = EstimationError::FixedLambdaNewtonDidNotConverge {
            context: "test Firth solve".to_string(),
            reason: FixedLambdaStallReason::LineSearchExhausted,
            objective_value: 3.25,
            stationarity: FixedLambdaStationarityEvidence {
                kind: FixedLambdaResidualKind::NewtonDecrement,
                residual: 0.125,
                bound: 1.0e-7,
            },
            checkpoint,
        };

        let display = err.to_string();
        assert!(display.contains("test Firth solve"));
        assert!(display.contains("line search exhausted"));
        assert!(display.contains("Newton decrement"));
        assert!(display.contains("2x1"));
        assert!(display.contains("11 iteration"));
        assert!(!display.contains("12345.6789"));
        assert!(!display.contains("98765.4321"));
        assert_eq!(
            format!("{err:?}"),
            display,
            "Debug delegates to safe Display"
        );
        assert!(err.is_inner_solve_retreat());
    }

    // ── From<LinalgError> ─────────────────────────────────────────────────────

    #[test]
    fn from_linalg_invalid_input_maps_to_invalid_input() {
        let linalg_err = LinalgError::InvalidInput("linalg msg".to_string());
        let err = EstimationError::from(linalg_err);
        assert!(matches!(err, EstimationError::InvalidInput(_)));
        assert!(err.to_string().contains("linalg msg"));
    }

    #[test]
    fn from_linalg_hessian_not_spd_maps_correctly() {
        let linalg_err = LinalgError::HessianNotPositiveDefinite {
            min_eigenvalue: -1.0,
        };
        let err = EstimationError::from(linalg_err);
        assert!(matches!(
            err,
            EstimationError::HessianNotPositiveDefinite { .. }
        ));
    }
}
