//! The penalized iteratively-reweighted-least-squares (P-IRLS) inner solver,
//! split by concern into real submodules. Each concern module pulls the shared
//! crate-level import surface and every sibling's items through a single
//! `use super::*;`, backed by the `pub(crate) use <mod>::*;` re-exports below —
//! reproducing the flat namespace the module historically shared, without any
//! line-cut `include!` fragments.

mod prelude;
pub use prelude::*;

// ── Concern modules ──────────────────────────────────────────────────────────
mod convergence;
mod curvature;
mod damping;
mod deviance;
mod dispersion;
mod edf;
mod family_state;
mod gam_working_model;
mod glm_update;
mod log_link_working_state;
pub(crate) mod loop_driver;
mod low_rank;
mod newton_solve;
mod penalty;
mod pls_solver;
mod reweight;
mod sparse_system;
mod state;
mod working_model_trait;
mod workspace;

#[cfg(test)]
mod tests;

// ── Concern-module re-export glue ────────────────────────────────────────────
// Glob re-exports keep each newly-split concern module's items reachable from
// `super::*` inside its siblings (reproducing the flat namespace the file shared
// before the split). The pre-existing real submodules below keep their explicit
// named re-exports so their exact visibility — and crate-facing `pub` API — is
// preserved; globbing them too would double-import every name listed there.
pub use curvature::*;
pub use deviance::*;
pub(crate) use dispersion::*;
pub use family_state::*;
pub(crate) use gam_working_model::*;
pub use glm_update::*;
pub use low_rank::*;
pub use newton_solve::*;
pub(crate) use sparse_system::*;
pub(crate) use working_model_trait::*;
pub use workspace::*;

// ── Pre-existing real-submodule re-exports (visibility preserved) ─────────────
use convergence::effective_kkt_tolerance;

use damping::{
    add_scaled_diagonal_to_upper_sparse, compute_lm_d2, update_scaled_diagonal_in_place,
};

pub use edf::StablePLSResult;

use edf::{
    calculate_edf_from_sparse_factor, calculate_edf_with_penalty,
    calculate_edfwithworkspace_from_factor, calculate_edfwithworkspace_with_penalty,
};

use log_link_working_state::ETA_CLAMP;

pub(crate) use penalty::PirlsPenalty;
use penalty::{
    KroneckerQsTransform, WorkingCoordinateDesign, WorkingReparamTransform, attach_penalty_shift,
};

use pls_solver::solve_penalized_least_squares_implicit;

pub use pls_solver::{GaussianFixedCache, SparseXtwxPrecomputed};
pub use sparse_system::{SparsePenalizedSystem, assemble_and_factor_sparse_penalized_system};

pub use reweight::runworking_model_pirls;

pub use state::array1_l2_norm;

// Surface the `WorkingModel` trait (defined in the private `working_model_trait`
// module) at the `pirls` root so out-of-crate engine implementors (gam-models
// survival working models) can name and implement it.
pub use working_model_trait::WorkingModel;

pub use state::{
    AdaptiveKktTolerance, ExportedLaplaceCurvature, FirthDiagnostics, HessianCurvatureKind,
    PirlsCoordinateFrame, PirlsLinearSolvePath, PirlsResult, PirlsStatus,
    WorkingModelIterationInfo, WorkingModelPirlsResult, WorkingState,
};

// loop_driver owns: default_beta_guess_external, solve_intercept_for_prevalence,
// assemble_pirls_result, detect_logit_instability, stack_lambdaweighted_penalty_root_canonical,
// build_sparse_native_reparam_result, build_diagonal_penalty_from_kronecker, canonical_prior_shift,
// canonical_prior_mean_aggregate, PirlsProblem, PenaltyConfig, fit_model_for_fixed_rho,
// fit_model_for_fixed_rho_with_adaptive_kkt, PirlsConfig, make_reparam_operator,
// build_transformed_lower_bound_constraints*, build_transformed_linear_constraints*,
// merge_linear_constraints, sparse_from_denseview.
use loop_driver::assert_symmetric_tol;

pub(crate) use loop_driver::fit_model_for_fixed_rho_with_adaptive_kkt;

pub use loop_driver::{PenaltyConfig, PirlsConfig, PirlsProblem, fit_model_for_fixed_rho};

/// Allow up to 128MB per thread for cached L-BFGS/PIRLS history.
pub(crate) const PIRLS_CACHE_BYTE_BUDGET: usize = 128 * 1024 * 1024;
