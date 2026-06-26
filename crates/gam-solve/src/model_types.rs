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

use ndarray::{Array1, Array2};

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

pub use gam_terms::penalty_spec::PenaltySpec;

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
