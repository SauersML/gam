//! Custom-family blockwise carrier, organized by concern.
//!
//! The carrier is split into real submodules, each a single defensible concern:
//!
//! - [`block_spec`]       — the coefficient group/label/prior data model plus the
//!   fit-level blockspec validator and the block-role heuristic. The block
//!   data-model types (`ParameterBlockSpec`, `ParameterBlockState`, …), the
//!   `CustomFamilyError`, the `PenaltyMatrix` carrier, and the
//!   internal-consistency validator now live in `gam-problem` (#1521); the
//!   `CustomFamily` trait, fit options, and ψ design-derivative operators live in
//!   `gam-model-api`. They are re-exported through the prelude below so the prior
//!   flat-namespace `crate::custom_family::*` API is unchanged.
//! - [`psi_design`]       — ψ design-derivative operators (now in `gam-model-api`).
//! - [`blockwise_solve`]  — the inner block-coordinate solve + numeric kernels.
//! - [`joint_newton`]     — joint (cross-block) Newton + trust region + PCG + KKT.
//! - [`outer_objective`]  — the outer (ρ) objective and inner-fit driver.
//! - [`assembly`], [`inner_blockwise_fit`], [`joint_derivatives`], [`warm_start`]
//!   — the #1145 split of `outer_objective`.
//! - [`psi_hyper`]        — ψ `HyperCoord` construction + hyper-objective eval.
//! - [`jeffreys`]         — the Jeffreys-prior contribution to the joint objective.
//! - [`covariance`]       — joint covariance/geometry + stationarity/KKT residuals.
//! - [`fit`]              — the public fit entry points + result assembly.
//! - [`penalty_labels`]   — penalty-label layout + labeled log-λ (de)aggregation.
//! - [`coefficient_groups`] — coefficient-group realization.
//! - [`custom_family_persistent_warm_start`] — the persistent warm-start cache
//!   (hosted at crate root since #1521; re-exported into the prelude here).
//!
//! Cross-submodule items are `pub(crate)`; each submodule pulls the shared
//! crate-internal imports below in via `use super::*;`.

// --- crate-internal (gam-solve) imports ---------------------------------
pub(crate) use crate::active_set::{
    project_stationarity_residual_on_constraint_cone, solve_quadratic_with_linear_constraints,
};
pub(crate) use crate::custom_family_persistent_warm_start::{
    capture_fit_artifact, consume_fit_artifact, load_persistent_custom_family_warm_start,
    store_persistent_custom_family_warm_start, update_custom_outer_inner_cap_from_warm_start,
};
pub(crate) use crate::estimate::reml::penalty_logdet::PenaltyPseudologdet;
pub(crate) use crate::estimate::reml::reml_outer_engine::{
    BlockCoupledOperator, CompositeHyperOperator, DenseSpectralOperator, DispersionHandling,
    ExactJeffreysTerm, HessianDerivativeProvider, HessianOperator, MatrixFreeSpdOperator,
    OuterHessianDerivativeKernel, PenaltySubspaceTrace, StochasticTraceState,
    compute_block_penalty_logdet_derivs, compute_efs_update, compute_hybrid_efs_update,
    exact_pseudo_logdet, hessian_operator_geometric_scale, positive_eigenvalue_threshold,
    spectral_epsilon, spectral_regularize,
};
// `ActiveLinearConstraintBlock`, `FitGeometry`, and `ProjectedKktResidual` are
// the model-estimation contract types that still live in the (not-yet-carved)
// crate-root `model_types` module; `EstimationError` already descended to
// `gam-problem` and arrives via the `gam_problem::*` glob below.
pub(crate) use crate::model_types::{
    ActiveLinearConstraintBlock, FitGeometry, ProjectedKktResidual,
};
pub(crate) use crate::pirls::solve_newton_directionwith_lower_bounds;

// --- lower-crate imports -------------------------------------------------
pub(crate) use faer::Side;
pub(crate) use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_atb, fast_av};
pub(crate) use gam_linalg::matrix::{
    DesignMatrix, EmbeddedColumnBlock, LinearOperator, SignedWeightsView, SymmetricMatrix,
    dense_rowwise_kronecker,
};
pub(crate) use gam_linalg::{RidgeDeterminantMode, RidgePolicy};
pub(crate) use gam_problem::finite_validation::{
    ensure_finite_scalar_estimation, validate_all_finite_estimation,
};
pub(crate) use gam_problem::{
    BlockLocalDrift, ContractedPsiSecondOrder, ContractedPsiSecondOrderFn,
    DenseMatrixHyperOperator, DriftDerivResult, FixedDriftDerivFn, HyperCoord, HyperCoordDrift,
    HyperCoordPair, HyperOperator,
};
pub(crate) use gam_problem::{LinearInequalityConstraints, PenaltyMatrix};
pub(crate) use gam_runtime::resource::{DerivativeStorageMode, ResourcePolicy};
pub(crate) use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, s};
pub(crate) use std::any::Any;
pub(crate) use std::cell::RefCell;
pub(crate) use std::collections::{BTreeMap, HashMap};
pub(crate) use std::ops::Range;
pub(crate) use std::sync::atomic::{AtomicUsize, Ordering};
pub(crate) use std::sync::{Arc, Mutex, OnceLock, Weak};
pub(crate) use thiserror::Error;

// --- descended carriers + the `CustomFamily` trait + fit options + ψ design --
// `pub use gam_problem::*` brings the block data model (`ParameterBlockSpec`,
// `ParameterBlockState`, working sets, channel-Hessians), `CustomFamilyError`,
// `BlockRole`, `RhoPrior`, `EvalMode`, `PseudoLogdetMode`,
// `validate_blockspec_consistency`, and the joint-penalty / psi-terms carriers.
// `pub use gam_model_api::families::custom_family::*` brings `CustomFamily`,
// `BlockwiseFitOptions`, `block_offsets_from_specs`, the outer-derivative
// policy/cost types, and the ψ design-derivative operators.
pub use gam_model_api::families::custom_family::*;
pub use gam_problem::*;

// `whitened_spectrum` is a submodule hosted inside `joint_newton`; re-export it
// at the carrier scope so the trust-region tests reach it through `super::*`.
pub(crate) use joint_newton::whitened_spectrum;

mod assembly;
mod block_spec;
mod blockwise_solve;
mod coefficient_groups;
mod covariance;
mod fit;
mod inner_blockwise_fit;
mod jeffreys;
mod joint_derivatives;
mod joint_newton;
mod outer_objective;
mod penalty_labels;
mod psi_hyper;
mod warm_start;

// `pub use ...::*` preserves each item's own visibility (pub stays pub,
// pub(crate) stays pub(crate)) so the prior flat-namespace API is unchanged.
pub use assembly::*;
pub use block_spec::*;
pub(crate) use blockwise_solve::*;
pub use coefficient_groups::*;
pub(crate) use covariance::*;
pub use fit::*;
pub(crate) use inner_blockwise_fit::*;
pub(crate) use jeffreys::*;
pub(crate) use joint_derivatives::*;
pub use joint_newton::*;
pub(crate) use outer_objective::*;
pub(crate) use penalty_labels::*;
pub use psi_hyper::*;
pub use warm_start::*;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
