use gam_linalg::LinalgError;
use gam_linalg::faer_ndarray::FaerLinalgError;
use serde::{Deserialize, Serialize};

use crate::{BasisError, CustomFamilyError, MonotoneRootError};

/// Which rung of the stationarity ladder produced the bound a refusal was
/// measured against (#2458).
///
/// A non-convergence refusal reports `|Pg|` against a bound, but routes reach
/// that bound by different machinery, and only one rung — the curvature
/// resolvability standard `√(2·h·τ)` — is derived from what a gradient of that
/// size does to the criterion. The rest are gradient-magnitude substitutes
/// adopted where the derived standard was unavailable. Without this field, "did
/// this route hold itself to the derived standard" is an inference from the
/// numbers rather than something the refusal states, which is what made the
/// duchon and #2479 adjudications read the ladder out of prose.
///
/// The rung enum itself lives in the solver that owns the ladder; this is the
/// neutral projection of it, so the problem layer carries the provenance
/// without importing a solver-specific vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StationarityRung {
    /// Stable rung label, e.g. `"curvature-resolvability"`.
    pub label: &'static str,
    /// Whether this rung is the derived resolvability standard rather than a
    /// gradient-magnitude substitute.
    pub derived_standard: bool,
}

impl StationarityRung {
    /// A zero-dimensional outer problem (#2530): no smoothing estimand exists,
    /// so the score is empty and exactly stationary *by construction* rather
    /// than by clearing any band. The `bound` on such a certificate is a
    /// formality — no ladder ran, because there was nothing to weigh — and
    /// borrowing a gradient rung for it would claim a comparison that never
    /// happened, which is the same error `NoComparison` exists to prevent one
    /// level down.
    pub const EMPTY_ESTIMAND: Self = Self {
        label: "empty-estimand",
        derived_standard: false,
    };
}

impl std::fmt::Display for StationarityRung {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rung={} derived_standard={}",
            self.label, self.derived_standard
        )
    }
}

/// What a non-convergence refusal was decided against (#2458/#2465).
///
/// A verdict is only falsifiable from the run record if it carries the quantity
/// it was decided against, and for a stationarity refusal that quantity is two
/// facts travelling together: the bound, and the rung that produced it. They
/// used to be two independent fields — `stationarity_bound: f64` beside
/// `stationarity_bound_rung: Option<StationarityRung>` — and the `Option` was a
/// hole any call site with a number to hand could take. Twenty of the outer
/// runner's thirty refusal paths took it.
///
/// Bundling the pair was not the whole defect. Most of those twenty refuse
/// *before any stationarity comparison exists*: a failed terminal evaluation, a
/// malformed gradient, a non-converged inner state. They reported the raw
/// configured tolerance — a constant the point was never weighed against — in a
/// sentence reading "projected gradient norm … against stationarity bound …",
/// asserting a pairing the code does not have. Naming that constant with a rung
/// makes the sentence *more* confident, not more honest; the fix is to report no
/// bound, because none was applied.
///
/// [`Self::Measured`] therefore means something specific and checkable: the
/// bound is a property of *this* point, derived from its own evidence by the
/// named rung, and the reported residual is the quantity weighed against it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
// `StationarityRung::label` is a `&'static str` — a fixed vocabulary, not run
// data — so a borrowing deserializer can only produce one from a `'static`
// input. Stating that here keeps the rung's stable-label representation rather
// than allocating a `String` per refusal to satisfy the derive.
#[serde(bound(deserialize = "'de: 'static"))]
pub enum StationarityStandard {
    /// A stationarity residual measured at this point was weighed against
    /// `bound`, which `rung` derived from this point's own evidence.
    Measured {
        bound: f64,
        rung: StationarityRung,
    },
    /// The refusal was decided without any stationarity comparison — the
    /// terminal evidence was rejected before a residual existed, or the
    /// predicate was an identity/existence check rather than a bound test. The
    /// refusal's `reason` carries the basis; no bound is reported because none
    /// was applied.
    NoComparison,
}

impl StationarityStandard {
    /// The bound, when one was applied.
    pub fn bound(&self) -> Option<f64> {
        match self {
            Self::Measured { bound, .. } => Some(*bound),
            Self::NoComparison => None,
        }
    }

    /// The rung that produced the bound, when one was applied.
    pub fn rung(&self) -> Option<StationarityRung> {
        match self {
            Self::Measured { rung, .. } => Some(*rung),
            Self::NoComparison => None,
        }
    }
}

impl std::fmt::Display for StationarityStandard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Measured { bound, rung } => {
                write!(f, "against stationarity bound {bound:.3e} ({rung})")
            }
            Self::NoComparison => f.write_str(
                "against no stationarity bound: this refusal was decided by the reason \
                 above, not by a stationarity comparison",
            ),
        }
    }
}

/// What an assembly-time convergence gate weighed (#2427/#2530).
///
/// A residual and the bound it was weighed against are one fact, not two. They
/// used to be two independent `Option<f64>` fields filled from DIFFERENT
/// sources with asymmetric fallbacks: the residual fell back to the exported
/// outer gradient norm when no certificate existed, the bound had no fallback.
/// A run with no certificate therefore rendered `stationarity residual
/// Some(0.0) against None` -- a residual weighed against nothing, printed as
/// though a comparison had happened, and recorded verbatim on #2471.
///
/// [`EstimationError::RemlDidNotConverge`] already forbids that shape one
/// variant up. This is the same repair here, and it REMOVES the fallback
/// rather than symmetrising it: if no certificate was assembled then no
/// first-order comparison was made, and substituting a norm of different
/// provenance is an absent measurement scored as a value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FitStationarityEvidence {
    /// A residual measured at the rejected point, weighed against `bound`.
    /// Both come from the SAME certificate, so they are one comparison.
    Certified { residual: f64, bound: f64 },
    /// No comparison was made: there was no certificate to take a residual and
    /// a bound from. The refusal's status strings carry the basis.
    NoComparison,
}

impl std::fmt::Display for FitStationarityEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certified { residual, bound } => {
                write!(f, "residual {residual:.3e} against bound {bound:.3e}")
            }
            Self::NoComparison => f.write_str("not compared: no certificate was assembled"),
        }
    }
}

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

/// The exact error thrown across a fatal outer-objective boundary.
///
/// Outer orchestration receives failures through two APIs. Direct evaluators
/// return [`EstimationError`], while optimizer-facing evaluators return
/// [`opt::ObjectiveEvalError`], which owns both the producer's recoverable/fatal
/// verdict and (when available) its typed source. Flattening the latter through
/// `into_message()` and minting a fresh `RemlOptimizationFailed` destroys that
/// source precisely where terminal classification and FFI dispatch need it.
///
/// This enum is the single owner of that distinction. The recursive
/// `EstimationError` arm is boxed for a finite representation; the optimizer
/// arm is retained whole, including its source chain and producer verdict.
#[derive(Debug, thiserror::Error)]
pub enum OuterObjectiveErrorSource {
    #[error(transparent)]
    Estimation(Box<EstimationError>),
    #[error(transparent)]
    Objective(opt::ObjectiveEvalError),
}

impl OuterObjectiveErrorSource {
    /// Recover an engine error without inspecting rendered prose.
    ///
    /// `ObjectiveEvalError` sources created by gam's objective bridge carry the
    /// originating `EstimationError` directly. A source owned by another
    /// optimizer client remains typed as that client's error and correctly
    /// returns `None`.
    #[must_use]
    pub fn estimation_error(&self) -> Option<&EstimationError> {
        match self {
            Self::Estimation(source) => Some(source),
            Self::Objective(source) => source.downcast_ref::<EstimationError>(),
        }
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
        "Beta precision refinement did not converge: after {passes} alternation pass(es) at the \
         selected smoothing the moment estimate moved from phi={prior_phi:.6e} to \
         phi={refreshed_phi:.6e} and the mean re-solve at the refreshed precision ended \
         '{inner_status}' (deviance {deviance:.6e}). The (beta, phi) alternation is only minted at \
         a fixed point where both the mean and the precision are stationary; a precision that \
         keeps growing means the response carries no dispersion around the fitted mean at this \
         smoothing, so no finite beta precision exists to certify."
    )]
    BetaPrecisionRefinementDidNotConverge {
        /// Alternation passes executed, counting the one whose re-solve failed.
        passes: usize,
        /// Precision the failed re-solve was warm-started from.
        prior_phi: f64,
        /// Precision the failed re-solve was asked to fit at.
        refreshed_phi: f64,
        /// Deviance of the last certified mean, at `prior_phi`.
        deviance: f64,
        /// Terminal status (or error) of the re-solve at `refreshed_phi`.
        inner_status: String,
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

    #[error("{}", hessian_not_positive_definite_message(*min_eigenvalue))]
    HessianNotPositiveDefinite { min_eigenvalue: f64 },

    #[error("REML smoothing optimization failed to converge: {0}")]
    RemlOptimizationFailed(String),

    /// A numerical refusal evaluated AT ONE TRIAL POINT of the outer smoothing
    /// search: no Laplace mode at this rho, an inner solve that missed its KKT
    /// bar at this rho, an indefinite trial Hessian at this rho.
    ///
    /// The outer search's response to this is to map the point to
    /// `OuterEval::infeasible` and step away — which is a normal thing for a
    /// lambda-search to consume, and the only response it *has*, since the only
    /// thing it can change is rho. Saying so in the type is the whole point:
    /// these refusals used to be reported as
    /// [`InvalidInput`](Self::InvalidInput) or `RemlOptimizationFailed`, both
    /// of which carry prose and both of which
    /// [`Self::is_trial_point_infeasible`] answers `false` for, so a correct
    /// per-rho verdict aborted the entire fit (#2531, #2590).
    ///
    /// It renders as the bare reason so a producer switching to it does not
    /// change the message a user or a regression test reads.
    #[error("{reason}")]
    TrialPointRefused { reason: String },

    #[error("Fatal outer-objective evaluation failure ({context}): {source}")]
    OuterObjectiveEvaluationFailed {
        context: String,
        #[source]
        source: OuterObjectiveErrorSource,
    },

    #[error(
        "Outer smoothing-parameter optimization did not certify a stationary optimum \
         ({context}): {reason} after {iterations} outer iteration(s); final objective \
         {final_value:.6e}, projected gradient norm {} {stationarity_standard}. A fit is \
         only minted from a converged optimization; the best iterate is carried as a \
         checkpoint — resume by seeding the outer search at rho_checkpoint = \
         {rho_checkpoint:?}.",
        .projected_grad_norm.map_or_else(|| "unmeasured".to_string(), |g| format!("{g:.3e}")),
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
        /// The standard this refusal was decided against: the bound together
        /// with the rung that produced it, or an explicit statement that no
        /// stationarity comparison was made (#2458/#2465). They are ONE field
        /// precisely so neither can be reported without the other, and so that
        /// a route which never formed a bound cannot print one.
        stationarity_standard: StationarityStandard,
        /// Best (lowest-objective feasible) outer iterate at exhaustion. This
        /// is work-preservation evidence for resume — it is NOT a fit and no
        /// fitted-model API is reachable from it.
        rho_checkpoint: Vec<f64>,
    },

    #[error(
        "Fit assembly rejected a non-converged optimization state: inner status \
         {inner_status}, outer status {outer_status}, after {outer_iterations} outer \
         iteration(s); final objective {}; stationarity {stationarity}, \
         step {step}. The best rho checkpoint is \
         {rho_checkpoint:?} and the resume token is {resume_token:?}; no fitted-model \
         API was constructed.",
        .final_value.map_or_else(
            || "unavailable (this fit has no criterion value)".to_string(),
            |value| format!("{value:.6e}"),
        ),
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
        /// Objective value at the best available checkpoint, or `None` when the
        /// rejected fit has no criterion value at all (the exact-fit Gaussian
        /// boundary). A refusal must not invent an objective it could not read.
        final_value: Option<f64>,
        /// The first-order residual together with the bound it was weighed
        /// against, or an explicit statement that no comparison was made.
        stationarity: FitStationarityEvidence,
        /// The accepted-step residual and its bound, same rule. Currently
        /// always `NoComparison` at the sole production site -- which is what
        /// the type should say, rather than leaving two independent `Option`s
        /// armed with the identical hazard for whoever wires them up.
        step: FitStationarityEvidence,
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
    /// Whether this failure invalidates the whole outer run or only the
    /// trial point it was produced at.
    ///
    /// The outer optimizer can survive an infeasible trial: it maps the
    /// point to `OuterEval::infeasible`, backs off, and continues. It
    /// cannot survive a structural failure. Deciding which is which is
    /// the producer's job, and the answer must travel with the error
    /// rather than be reconstructed downstream from its rendered text
    /// (#2553).
    ///
    /// Only failures that are genuinely a property of *this theta* answer
    /// `true`. Everything else stays fatal, which is the conservative
    /// direction: misclassifying a structural failure as recoverable
    /// would let the search grind through a problem that can never work.
    /// The match is deliberately exhaustive with no wildcard arm, for the
    /// same reason [`CustomFamilyError::is_trial_point_infeasible`] is: under
    /// a `_ => false` a newly added variant is classified *fatal* by the
    /// absence of a decision, and whoever adds it is never asked. That is how
    /// a rho-local refusal reached this function as `RemlOptimizationFailed` —
    /// a variant that carries only prose — and aborted a fit the outer search
    /// was equipped to walk away from (#2590).
    #[must_use]
    pub fn is_trial_point_infeasible(&self) -> bool {
        match self {
            // The producer classified it; ask it (#2553).
            Self::CustomFamily(err) => err.is_trial_point_infeasible(),
            // The producer said so directly (#2531).
            Self::TrialPointRefused { .. } => true,
            // "The inner problem at THIS rho is too hard to evaluate, try a
            // different rho" — [`Self::is_inner_solve_retreat`]'s own words for
            // exactly these five, and verbatim this predicate's definition. The
            // two used to disagree, so a P-IRLS budget exhaustion was a retreat
            // at one layer and a fatal at this one; #2593 unified them, and
            // `is_inner_solve_retreat` now reads this table rather than keeping
            // a second one.
            Self::ModelIsIllConditioned { .. }
            | Self::PerfectSeparationDetected { .. }
            | Self::MultinomialSeparationDetected { .. }
            | Self::PirlsDidNotConverge { .. }
            | Self::FixedLambdaNewtonDidNotConverge { .. } => true,
            // A structural failure, an already-terminal outer verdict, or a
            // statement about the configuration, the data, or the prediction
            // request: none of these becomes true or false by moving rho.
            Self::InvalidStabilization { .. }
            | Self::BasisError { .. }
            | Self::LinearSystemSolveFailed { .. }
            | Self::EigendecompositionFailed { .. }
            | Self::PenaltySpectrumNonFinite { .. }
            | Self::PenaltySpectrumIndefinite { .. }
            | Self::ParameterConstraintViolation { .. }
            | Self::BlockOrthogonalRemlDidNotConverge { .. }
            | Self::NegativeBinomialAlternationDidNotConverge { .. }
            | Self::BetaPrecisionRefinementDidNotConverge { .. }
            | Self::PrefitPerfectSeparationDetected { .. }
            | Self::PrefitLinearSeparationDetected { .. }
            | Self::PrefitRankDeficientDesignDetected { .. }
            | Self::PrefitNearDegenerateDesignDetected { .. }
            | Self::HessianNotPositiveDefinite { .. }
            | Self::RemlOptimizationFailed { .. }
            | Self::OuterObjectiveEvaluationFailed { .. }
            | Self::RemlDidNotConverge { .. }
            | Self::FitDidNotConverge { .. }
            | Self::GradientUnavailable { .. }
            | Self::LayoutError { .. }
            | Self::InvalidInput { .. }
            | Self::InverseLinkDomainViolation { .. }
            | Self::PirlsRowGeometryUnrepresentable { .. }
            | Self::ExactTweedieSeriesWorkLimit { .. }
            | Self::LogStrengthDomainViolation { .. }
            | Self::MonotoneRoot { .. }
            | Self::CalibratorTrainingFailed { .. }
            | Self::InvalidSpecification { .. }
            | Self::PredictionError { .. } => false,
        }
    }

    /// Preserve a thrown outer-objective failure across seed, solver, and
    /// fallback-plan orchestration. Trial-domain refusals must be represented
    /// as a finite API outcome (`+inf` / `OuterEval::infeasible`); an `Err`
    /// means the evaluation artifact itself could not be constructed and must
    /// never be retried as another numerical point.
    pub fn fatal_outer_evaluation(context: impl Into<String>, source: EstimationError) -> Self {
        if matches!(
            &source,
            EstimationError::OuterObjectiveEvaluationFailed { .. }
        ) {
            source
        } else {
            EstimationError::OuterObjectiveEvaluationFailed {
                context: context.into(),
                source: OuterObjectiveErrorSource::Estimation(Box::new(source)),
            }
        }
    }

    /// Preserve an optimizer-facing fatal evaluator failure without reminting
    /// its message as an unrelated [`Self::RemlOptimizationFailed`].
    ///
    /// The caller must have already consumed recoverable failures as rejected
    /// trial points. Requiring the producer's fatal verdict here makes an
    /// accidental promotion fail at the boundary that attempted it instead of
    /// silently changing control flow.
    pub fn fatal_objective_evaluation(
        context: impl Into<String>,
        source: opt::ObjectiveEvalError,
    ) -> Self {
        assert!(
            source.is_fatal(),
            "fatal_objective_evaluation requires a producer-classified fatal error"
        );
        EstimationError::OuterObjectiveEvaluationFailed {
            context: context.into(),
            source: OuterObjectiveErrorSource::Objective(source),
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
    /// Re-report this failure with more context WITHOUT changing whether it
    /// is a trial-point refusal.
    ///
    /// A wrapper that renders its source into a string and then picks a fresh
    /// variant silently overwrites the producer's verdict. That is how a
    /// per-rho survival-LAML stationarity refusal reached the outer boundary
    /// as `InvalidInput` and killed the fit (#2531), and how a typed
    /// `InnerSolveNotConverged` reached it as `RemlOptimizationFailed` and did
    /// the same (#2590). Any site that adds context to an error it did not
    /// produce should use this instead of choosing a variant for it.
    #[must_use]
    pub fn wrap_preserving_trial_point(self, context: &str) -> Self {
        let infeasible = self.is_trial_point_infeasible();
        let reason = format!("{context}: {self}");
        if infeasible {
            Self::TrialPointRefused { reason }
        } else {
            Self::InvalidInput(reason)
        }
    }

    pub fn is_inner_solve_retreat(&self) -> bool {
        // ONE table. This method and `is_trial_point_infeasible` ask the same
        // question -- "is this a statement about this rho, or about the
        // problem?" -- and used to answer it from two separate variant lists
        // that had drifted apart. Keeping a second list here is what let them
        // drift, so there is no longer a second list (#2593).
        //
        // The relation is delegation, not equality: `is_trial_point_infeasible`
        // is strictly wider, because it also asks the producer through
        // `CustomFamily(..)` and honours the typed `TrialPointRefused`. Every
        // retreat is an infeasibility; not every infeasibility arrives as one
        // of the five inner-solve shapes.
        self.is_trial_point_infeasible()
    }
}

#[cfg(test)]
mod trial_point_classification_tests {
    use super::*;

    /// The two classifiers now agree by construction, and this is what pins
    /// that: every inner-solve retreat is a trial-point infeasibility. The test
    /// it replaces pinned the opposite, deliberately, so that unifying them had
    /// to be a decision rather than a drift (#2593).
    #[test]
    fn every_inner_solve_retreat_is_a_trial_point_infeasibility() {
        let retreats = [
            EstimationError::ModelIsIllConditioned {
                condition_number: 1.0e18,
            },
            EstimationError::PerfectSeparationDetected {
                iteration: 3,
                max_abs_eta: 1.0e3,
            },
            EstimationError::MultinomialSeparationDetected {
                iteration: 3,
                max_abs_eta: 1.0e3,
                active_class_index: 1,
                row_index: 2,
            },
            EstimationError::PirlsDidNotConverge {
                max_iterations: 40,
                last_change: 1.0e-2,
            },
            // The fifth shape. The fixture carried four while the table it
            // covers -- and this test's own doc, five lines up -- says five, so
            // the fixed-lambda Newton stall was the one arm nothing pinned.
            // That is the retreat the multinomial and independent-binomial
            // vector-GLM lanes actually emit, and it is the arm a regression
            // flipping to `false` would have slipped past unnoticed.
            EstimationError::FixedLambdaNewtonDidNotConverge {
                context: "trial-point classification fixture".to_string(),
                reason: FixedLambdaStallReason::IterationBudgetExhausted,
                objective_value: 12.5,
                stationarity: FixedLambdaStationarityEvidence {
                    kind: FixedLambdaResidualKind::PenalizedGradientNorm,
                    residual: 1.0e-3,
                    bound: 1.0e-8,
                },
                checkpoint: FixedLambdaCheckpoint::new(
                    FixedLambdaSolverStage::MultinomialNewton,
                    vec![0.0, 0.0],
                    2,
                    1,
                    40,
                )
                .expect("fixture checkpoint geometry is valid"),
            },
        ];
        for error in retreats {
            assert!(
                error.is_inner_solve_retreat(),
                "fixture must be a retreat: {error}"
            );
            assert!(
                error.is_trial_point_infeasible(),
                "a retreat is by its own definition a trial-point infeasibility: {error}"
            );
        }
    }

    /// The failure #2590 is about: a refusal produced at one rho must not be
    /// graded fatal because it crossed a boundary that kept only its text.
    #[test]
    fn a_custom_family_trial_point_refusal_stays_recoverable() {
        let reason = "joint Newton returned an indefinite mode at this rho";
        assert!(
            EstimationError::CustomFamily(CustomFamilyError::trial_point(reason))
                .is_trial_point_infeasible()
        );
        assert!(
            !EstimationError::RemlOptimizationFailed(reason.to_string())
                .is_trial_point_infeasible(),
            "the prose-only variant is exactly what must NOT carry a rho-local refusal"
        );
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

    // ── stationarity rung provenance (#2458) ─────────────────────────────────

    fn reml_refusal(standard: StationarityStandard) -> EstimationError {
        EstimationError::RemlDidNotConverge {
            context: "unit".to_string(),
            reason: "budget exhausted".to_string(),
            iterations: 7,
            final_value: -1.25,
            projected_grad_norm: Some(7.5e-1),
            stationarity_standard: standard,
            rho_checkpoint: vec![0.5],
        }
    }

    fn measured(label: &'static str, derived_standard: bool) -> StationarityStandard {
        StationarityStandard::Measured {
            bound: 1.0e-2,
            rung: StationarityRung {
                label,
                derived_standard,
            },
        }
    }

    /// The whole point of the increment: a red states which standard it was
    /// held to, so a reader does not have to infer the rung from the numbers.
    #[test]
    fn refusal_message_carries_the_rung_and_whether_it_is_derived() {
        let derived = reml_refusal(measured("curvature-resolvability", true)).to_string();
        assert!(
            derived.contains("rung=curvature-resolvability"),
            "refusal must name its rung: {derived}"
        );
        assert!(
            derived.contains("derived_standard=true"),
            "refusal must say whether the rung is the derived standard: {derived}"
        );

        let substitute = reml_refusal(measured("solver-band", false)).to_string();
        assert!(substitute.contains("rung=solver-band"), "{substitute}");
        assert!(
            substitute.contains("derived_standard=false"),
            "a gradient-magnitude substitute must not read as the derived standard: {substitute}"
        );
    }

    /// A refusal reached before any stationarity comparison must not print a
    /// bound. The old shape filled the field with the raw configured tolerance
    /// and rendered "projected gradient norm … against stationarity bound …",
    /// which reads as a comparison that never happened — and naming the
    /// constant with a rung only made the false sentence more confident.
    #[test]
    fn a_refusal_without_a_comparison_reports_no_bound() {
        let message = reml_refusal(StationarityStandard::NoComparison).to_string();
        assert!(
            message.contains("against no stationarity bound"),
            "a refusal that applied no bound must say so: {message}"
        );
        assert!(
            !message.contains("rung="),
            "no rung may be claimed where no bound was applied: {message}"
        );
        assert!(
            !message.contains("1.000e-2"),
            "no bound value may appear where none was applied: {message}"
        );
    }

    /// The carrier's whole purpose: a bound cannot be read without the rung that
    /// produced it, and a route with neither reports neither.
    #[test]
    fn the_bound_and_its_rung_are_one_field() {
        let standard = measured("probe-noise-floor", false);
        assert_eq!(standard.bound(), Some(1.0e-2));
        assert_eq!(
            standard.rung().map(|rung| rung.label),
            Some("probe-noise-floor")
        );
        assert_eq!(StationarityStandard::NoComparison.bound(), None);
        assert_eq!(StationarityStandard::NoComparison.rung(), None);
    }

    /// The bound itself still reaches the message unchanged — the rung rides
    /// alongside it, it does not replace it.
    #[test]
    fn rung_rides_beside_the_bound_without_displacing_it() {
        let message = reml_refusal(measured("solver-band", false)).to_string();
        assert!(
            message.contains("1.000e-2"),
            "bound must survive: {message}"
        );
        assert!(
            message.contains("7.500e-1"),
            "projected gradient norm must survive: {message}"
        );
    }

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

/// Honest failure text for [`EstimationError::HessianNotPositiveDefinite`].
///
/// A failed Cholesky with a strictly POSITIVE reported minimum eigenvalue is
/// not an indefinite matrix — it is a positive spectrum whose condition
/// number exceeds float precision (the pivots collapse under roundoff), or a
/// non-finite assembly. Saying "not positive definite (minimum eigenvalue:
/// 4.1e1)" sent debugging at the wrong defect class (#2316 triage), so the
/// message now names the regime the eigenvalue actually indicates.
fn hessian_not_positive_definite_message(min_eigenvalue: f64) -> String {
    if min_eigenvalue.is_finite() && min_eigenvalue > 0.0 {
        format!(
            "Hessian factorization failed although the (lower-triangle) spectrum is positive \
             (minimum eigenvalue: {min_eigenvalue:.4e}): the condition number exceeds float \
             precision or the assembled matrix is asymmetric/non-finite outside the factored \
             triangle. This indicates a numerical instability in the Hessian assembly or scaling."
        )
    } else {
        format!(
            "Hessian matrix is not positive definite (minimum eigenvalue: {min_eigenvalue:.4e}). \
             This indicates a numerical instability."
        )
    }
}
