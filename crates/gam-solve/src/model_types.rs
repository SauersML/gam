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

use ndarray::{Array1, Array2, s};

pub use gam_problem::EstimationError;

// ===========================================================================
// Dispersion
// ===========================================================================

/// Dispersion contract used by inferential covariance and reference distributions.
///
/// This type lives in `gam-problem`; re-exported here so model result APIs and
/// existing engine code name the same neutral contract.
pub use gam_problem::Dispersion;

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
/// This type now lives in the neutral `gam-problem` crate (with its inherent
/// `evaluate` returning `gam_problem::PriorMeanError`); re-exported here so all
/// existing `crate::estimate::CoefficientPriorMean` references keep resolving.
/// Solver-side callers map `PriorMeanError` into `EstimationError::InvalidInput`.
pub use gam_problem::CoefficientPriorMean;

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
        structure_hint: Option<gam_terms::smooth::PenaltyStructureHint>,
        /// Optional operator-form handle bit-equivalent to `local`.
        op: Option<std::sync::Arc<dyn gam_terms::analytic_penalties::PenaltyOp>>,
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
    pub fn op(&self) -> Option<&std::sync::Arc<dyn gam_terms::analytic_penalties::PenaltyOp>> {
        match self {
            PenaltySpec::Block { op, .. } => op.as_ref(),
            PenaltySpec::Dense(_) | PenaltySpec::DenseWithMean { .. } => None,
        }
    }

    /// Convert from a `BlockwisePenalty`, preserving the structure hint and op.
    pub fn from_blockwise(bp: gam_terms::smooth::BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local,
            col_range: bp.col_range,
            prior_mean: bp.prior_mean,
            structure_hint: bp.structure_hint,
            op: bp.op,
        }
    }

    pub fn from_blockwise_ref(bp: &gam_terms::smooth::BlockwisePenalty) -> Self {
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

pub(crate) mod result_types;

pub(crate) use result_types::CERTIFICATE_RAIL_MARGIN;
pub use result_types::{
    AdaptiveRegularizationOptions, BlockRole, CriterionCertificate, ExecutionPath, FitArtifacts,
    FitGeometry, FitInference, FitOptions, FittedBlock, FittedLinkState, UnifiedFitResult,
    UnifiedFitResultParts, ensure_finite_scalar, saved_latent_cloglog_state_from_fit,
    saved_mixture_state_from_fit, saved_sas_state_from_fit, validate_all_finite,
    validate_dense_hessian_export, validate_explicit_dense_hessian_for_whitening,
};
pub(crate) use result_types::{ensure_finite_scalar_estimation, validate_all_finite_estimation};
