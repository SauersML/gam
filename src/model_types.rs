//! # Shared model-estimation contract types
//!
//! Lower-layer types that both the `families` layer (which constructs penalty
//! and dispersion specifications and propagates estimation errors) and the
//! `solver` layer (which consumes them) need to name. Hosting them here breaks
//! the `families → solver::estimate` back-edge that #1135 tracks: families now
//! import these from `crate::model_types` instead of reaching *up* into
//! `crate::solver::estimate`.
//!
//! ## Layering
//! These types depend only on lower or sibling layers (`linalg`, `terms`,
//! `families`' error types) — never on `solver`. `EstimationError` carries
//! `#[from]` conversions for the family error types it wraps, which is the
//! allowed downward direction.

use std::ops::Range;
use std::sync::Arc;

use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};

use crate::faer_ndarray::FaerLinalgError;

// ===========================================================================
// EstimationError
// ===========================================================================

/// A comprehensive error type for the model estimation process.
#[derive(thiserror::Error)]
pub enum EstimationError {
    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] crate::basis::BasisError),

    #[error("Custom-family fit failed: {0}")]
    CustomFamily(#[from] crate::families::custom_family::CustomFamilyError),

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

    #[error("monotone root solve: {0}")]
    MonotoneRoot(#[from] crate::families::monotone_root::MonotoneRootError),

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
    /// Classifies inner-solve failures that the outer REML loop should
    /// treat as a soft retreat (return +inf cost / infeasible outer-eval)
    /// rather than propagate as a hard error.
    ///
    /// Why: when the penalised Hessian becomes effectively singular at the
    /// current ρ, when P-IRLS hits a perfect-separation diagnostic, or when
    /// it exhausts its iteration budget, the outer optimiser's correct
    /// response is to back away from this ρ — not to terminate the fit.
    /// All three variants encode "the inner problem at this ρ is too hard
    /// to evaluate, try a different ρ".
    pub fn is_inner_solve_retreat(&self) -> bool {
        matches!(
            self,
            EstimationError::ModelIsIllConditioned { .. }
                | EstimationError::PerfectSeparationDetected { .. }
                | EstimationError::MultinomialSeparationDetected { .. }
                | EstimationError::PirlsDidNotConverge { .. }
        )
    }
}

impl From<crate::linalg::LinalgError> for EstimationError {
    fn from(error: crate::linalg::LinalgError) -> Self {
        match error {
            crate::linalg::LinalgError::InvalidInput(message) => {
                EstimationError::InvalidInput(message)
            }
            crate::linalg::LinalgError::HessianNotPositiveDefinite { min_eigenvalue } => {
                EstimationError::HessianNotPositiveDefinite { min_eigenvalue }
            }
            crate::linalg::LinalgError::ModelIsIllConditioned { condition_number } => {
                EstimationError::ModelIsIllConditioned { condition_number }
            }
        }
    }
}

// ===========================================================================
// Dispersion
// ===========================================================================

/// Dispersion contract used by inferential covariance and reference distributions.
///
/// `Known(phi)` is used for fixed-scale exponential-family fits such as
/// Poisson and Binomial (`phi = 1`). `Estimated(phi)` is used when the
/// residual/likelihood scale is estimated from the data, e.g. Gaussian
/// (`phi = sigma^2`) and Gamma (`phi = 1 / shape`). Stored covariance
/// matrices below are scaled by this `phi`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Dispersion {
    Known(f64),
    Estimated(f64),
}

impl Dispersion {
    #[inline]
    pub const fn phi(self) -> f64 {
        match self {
            Self::Known(phi) | Self::Estimated(phi) => phi,
        }
    }

    #[inline]
    pub const fn is_estimated(self) -> bool {
        matches!(self, Self::Estimated(_))
    }
}

impl Default for Dispersion {
    fn default() -> Self {
        Self::Known(1.0)
    }
}

// ===========================================================================
// Constraint/KKT carriers
// ===========================================================================

/// Active row block of the joint linear inequality constraint matrix at the
/// converged inner iterate.
#[derive(Clone, Debug)]
pub struct ActiveLinearConstraintBlock {
    /// `k_active x p` matrix of active constraint rows.
    pub a: Array2<f64>,
}

/// Subspace represented by a stored KKT residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KktResidualSubspace {
    /// Residual after active-constraint normal components have been stripped:
    /// `r_A = P_T(Sbeta + Gamma beta - grad ell)`.
    ActiveProjected,
    /// Residual additionally projected into the retained identifiable range:
    /// `r_R = R R^T r_A`.
    ReducedRange,
}

/// KKT residual `r = grad_beta L_pen(beta_hat)` at the converged inner
/// iterate, tagged with the exact represented subspace.
#[derive(Clone, Debug)]
pub struct ProjectedKktResidual {
    /// The residual vector in the full coefficient coordinates. Active and
    /// reduced-range projection zero out excluded directions rather than
    /// shortening the vector, so its length remains `p`.
    pub(crate) residual: Array1<f64>,
    pub(crate) subspace: KktResidualSubspace,
    /// The KKT-stationarity tolerance the inner solver compared the residual
    /// against when the certificate fired.
    pub(crate) residual_tol: Option<f64>,
    /// `total_p - active_set_size` at the producing iterate.
    pub(crate) free_rank: Option<usize>,
}

impl ProjectedKktResidual {
    /// Construct from `r_A = P_T(Sbeta + Gamma beta - grad ell)`, with active
    /// constraint multipliers removed but before any reduced-range projection.
    pub(crate) fn from_active_projected(residual: Array1<f64>) -> Self {
        Self {
            residual,
            subspace: KktResidualSubspace::ActiveProjected,
            residual_tol: None,
            free_rank: None,
        }
    }

    /// Construct from `r_R = R R^T r_A`, where `R` is the actual reduced
    /// identifiable basis used by the projected inverse kernel.
    pub(crate) fn from_reduced_range(residual: Array1<f64>) -> Self {
        Self {
            residual,
            subspace: KktResidualSubspace::ReducedRange,
            residual_tol: None,
            free_rank: None,
        }
    }

    /// Attach the KKT tolerance and free-subspace rank to a previously
    /// constructed residual.
    pub(crate) fn with_metadata(mut self, residual_tol: f64, free_rank: usize) -> Self {
        self.residual_tol = Some(residual_tol);
        self.free_rank = Some(free_rank);
        self
    }

    /// Borrow the underlying free-space residual for the H^-1*r solve and its
    /// rho-derivatives.
    pub fn as_array(&self) -> &Array1<f64> {
        &self.residual
    }

    pub fn subspace(&self) -> KktResidualSubspace {
        self.subspace
    }
}

// ===========================================================================
// CoefficientPriorMean + PenaltySpec
// ===========================================================================

/// Programmatic prior mean for a coefficient penalty block.
///
/// The mean is evaluated once during penalty canonicalization and then enters
/// the solver as the centering vector in `(beta - mean)' S (beta - mean)`.
#[derive(Clone, Default)]
pub enum CoefficientPriorMean {
    #[default]
    Zero,
    Scalar(f64),
    Constant(Array1<f64>),
    Functional {
        metadata: Array1<f64>,
        evaluator: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    },
    /// Covariate-functional mean `mu(a) = amplitude * K(a)` for a coefficient block.
    ///
    /// Formula-level coefficient groups pass their row/covariate metadata as
    /// `covariates`; the user-supplied kernel returns the block-sized basis
    /// vector `K(a)` and the scalar amplitude supplies `alpha`.
    KernelBasis {
        covariates: Array1<f64>,
        amplitude: f64,
        kernel: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    },
}

impl std::fmt::Debug for CoefficientPriorMean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero => f.write_str("Zero"),
            Self::Scalar(value) => f.debug_tuple("Scalar").field(value).finish(),
            Self::Constant(values) => f
                .debug_tuple("Constant")
                .field(&format_args!("len={}", values.len()))
                .finish(),
            Self::Functional { metadata, .. } => f
                .debug_struct("Functional")
                .field("metadata_len", &metadata.len())
                .finish_non_exhaustive(),
            Self::KernelBasis {
                covariates,
                amplitude,
                ..
            } => f
                .debug_struct("KernelBasis")
                .field("covariate_len", &covariates.len())
                .field("amplitude", amplitude)
                .finish_non_exhaustive(),
        }
    }
}

impl CoefficientPriorMean {
    pub const fn scalar(value: f64) -> Self {
        Self::Scalar(value)
    }

    pub fn constant(values: Array1<f64>) -> Self {
        Self::Constant(values)
    }

    pub fn functional(
        metadata: Array1<f64>,
        evaluator: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    ) -> Self {
        Self::Functional {
            metadata,
            evaluator,
        }
    }

    pub fn kernel_basis(
        covariates: Array1<f64>,
        amplitude: f64,
        kernel: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    ) -> Self {
        Self::KernelBasis {
            covariates,
            amplitude,
            kernel,
        }
    }

    pub(crate) fn evaluate(
        &self,
        block_dim: usize,
        context: &str,
    ) -> Result<Array1<f64>, EstimationError> {
        let values = match self {
            Self::Zero => Array1::zeros(block_dim),
            Self::Scalar(value) => {
                if !value.is_finite() {
                    crate::bail_invalid_estim!(
                        "{context}: coefficient prior mean scalar must be finite, got {value}"
                    );
                }
                Array1::from_elem(block_dim, *value)
            }
            Self::Constant(values) => values.clone(),
            Self::Functional {
                metadata,
                evaluator,
            } => evaluator(metadata),
            Self::KernelBasis {
                covariates,
                amplitude,
                kernel,
            } => {
                if !amplitude.is_finite() {
                    crate::bail_invalid_estim!(
                        "{context}: coefficient prior mean amplitude must be finite, got {amplitude}"
                    );
                }
                let mut values = kernel(covariates);
                values *= *amplitude;
                values
            }
        };
        if values.len() != block_dim {
            crate::bail_invalid_estim!(
                "{context}: coefficient prior mean length must be {block_dim}, got {}",
                values.len()
            );
        }
        if values.iter().any(|&value| !value.is_finite()) {
            crate::bail_invalid_estim!(
                "{context}: coefficient prior mean contains non-finite values"
            );
        }
        Ok(values)
    }
}

/// A penalty specification for the public estimate API.
///
/// `Block` stores only the active sub-block and its column range, avoiding
/// the O(p^2) cost of embedding into a full penalty matrix.
/// `Dense` stores a full `p x p` penalty matrix for callers that already
/// have one.
#[derive(Clone)]
pub enum PenaltySpec {
    /// Block-local penalty: `local` is `block_dim x block_dim`,
    /// applied to columns `col_range` of the coefficient vector.
    Block {
        local: Array2<f64>,
        col_range: Range<usize>,
        prior_mean: CoefficientPriorMean,
        /// Optional structural hint for fast-path spectral decomposition.
        structure_hint: Option<crate::terms::smooth::PenaltyStructureHint>,
        /// Optional operator-form handle bit-equivalent to `local`.
        op: Option<std::sync::Arc<dyn crate::terms::analytic_penalties::PenaltyOp>>,
    },
    /// Full dense penalty matrix (`p x p`).
    Dense(Array2<f64>),
    /// Full dense penalty matrix with a programmatic prior mean in the same
    /// global coefficient basis.
    DenseWithMean {
        matrix: Array2<f64>,
        prior_mean: CoefficientPriorMean,
    },
}

impl std::fmt::Debug for PenaltySpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PenaltySpec::Block {
                local,
                col_range,
                prior_mean,
                structure_hint,
                op,
            } => f
                .debug_struct("Block")
                .field(
                    "local",
                    &format_args!("{}×{}", local.nrows(), local.ncols()),
                )
                .field("col_range", col_range)
                .field("prior_mean", prior_mean)
                .field("structure_hint", structure_hint)
                .field("op", &op.as_ref().map(|o| o.dim()))
                .finish(),
            PenaltySpec::Dense(m) => f
                .debug_tuple("Dense")
                .field(&format_args!("{}×{}", m.nrows(), m.ncols()))
                .finish(),
            PenaltySpec::DenseWithMean { matrix, prior_mean } => f
                .debug_struct("DenseWithMean")
                .field(
                    "matrix",
                    &format_args!("{}×{}", matrix.nrows(), matrix.ncols()),
                )
                .field("prior_mean", prior_mean)
                .finish(),
        }
    }
}

impl PenaltySpec {
    /// The column range this penalty covers.
    /// For `Dense`, this is `0..p` where `p = m.ncols()`.
    pub fn col_range(&self, p: usize) -> Range<usize> {
        match self {
            PenaltySpec::Block { col_range, .. } => col_range.clone(),
            PenaltySpec::Dense(m) => {
                assert_eq!(m.ncols(), p);
                0..p
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                assert_eq!(matrix.ncols(), p);
                0..p
            }
        }
    }

    /// Op-form handle when present (only for `Block`; `Dense` always returns `None`).
    pub fn op(&self) -> Option<&std::sync::Arc<dyn crate::terms::analytic_penalties::PenaltyOp>> {
        match self {
            PenaltySpec::Block { op, .. } => op.as_ref(),
            PenaltySpec::Dense(_) | PenaltySpec::DenseWithMean { .. } => None,
        }
    }

    /// Convert from a `BlockwisePenalty`, preserving the structure hint and op.
    pub fn from_blockwise(bp: crate::terms::smooth::BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local,
            col_range: bp.col_range,
            prior_mean: bp.prior_mean,
            structure_hint: bp.structure_hint,
            op: bp.op,
        }
    }

    pub fn from_blockwise_ref(bp: &crate::terms::smooth::BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local.clone(),
            col_range: bp.col_range.clone(),
            prior_mean: bp.prior_mean.clone(),
            structure_hint: bp.structure_hint.clone(),
            op: bp.op.clone(),
        }
    }

    /// Materialize the full `p x p` dense penalty matrix.
    /// For `Dense`, this is a clone.  For `Block`, this embeds `local` into a
    /// zero matrix at the given `col_range`.
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            PenaltySpec::Dense(m) => m.clone(),
            PenaltySpec::DenseWithMean { matrix, .. } => matrix.clone(),
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let p = col_range.end.max(local.nrows());
                // Caller should supply p externally when the total dim is larger;
                // this is the best we can do without it.
                let mut out = Array2::zeros((p, p));
                out.slice_mut(s![col_range.clone(), col_range.clone()])
                    .assign(local);
                out
            }
        }
    }

    /// Materialize the full `p_total x p_total` dense penalty matrix.
    /// For `Dense`, this is a clone (asserts that it matches `p_total`).
    /// For `Block`, this embeds `local` into a `p_total x p_total` zero matrix.
    pub fn to_global(&self, p_total: usize) -> Array2<f64> {
        match self {
            PenaltySpec::Dense(m) => {
                assert_eq!(m.nrows(), p_total);
                m.clone()
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                assert_eq!(matrix.nrows(), p_total);
                matrix.clone()
            }
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let mut out = Array2::zeros((p_total, p_total));
                out.slice_mut(s![col_range.clone(), col_range.clone()])
                    .assign(local);
                out
            }
        }
    }
}

mod result_types;

pub use result_types::{
    AdaptiveRegularizationOptions, BlockRole, CriterionCertificate, FitArtifacts, FitGeometry,
    FitInference, FitOptions, FittedBlock, FittedLinkState, UnifiedFitResult,
    UnifiedFitResultParts, ensure_finite_scalar, saved_latent_cloglog_state_from_fit,
    saved_mixture_state_from_fit, saved_sas_state_from_fit, validate_all_finite,
    validate_dense_hessian_export, validate_explicit_dense_hessian_for_whitening,
};
pub(crate) use result_types::{
    CERTIFICATE_RAIL_MARGIN, CERTIFICATE_Z_GATE,
};
pub(crate) use result_types::{ensure_finite_scalar_estimation, validate_all_finite_estimation};
