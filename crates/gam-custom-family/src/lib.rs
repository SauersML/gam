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
//!   flat-namespace `crate::*` API is unchanged.
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

// The `#[macro_export]` error-bail macros live in `gam-problem` (its crate
// root). Importing them here makes `crate::bail_invalid_estim!` /
// `crate::bail_dim_custom!` resolve at every call site exactly as they did when
// this carrier lived in `gam-solve` (whose `mod.rs` performed the same import).
pub(crate) use gam_problem::{bail_dim_custom, bail_invalid_estim};

// --- crate-internal (gam-solve) imports ---------------------------------
pub(crate) use gam_solve::active_set::{
    project_stationarity_residual_on_constraint_cone, solve_quadratic_with_linear_constraints,
};
pub(crate) use crate::custom_family_persistent_warm_start::{
    capture_fit_artifact, consume_fit_artifact, load_persistent_custom_family_warm_start,
    store_persistent_custom_family_warm_start, update_custom_outer_inner_cap_from_warm_start,
};
pub(crate) use gam_solve::estimate::reml::penalty_logdet::PenaltyPseudologdet;
pub(crate) use gam_solve::estimate::reml::reml_outer_engine::{
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
pub(crate) use gam_solve::model_types::{
    ActiveLinearConstraintBlock, FitGeometry, ProjectedKktResidual,
};
pub(crate) use gam_solve::pirls::solve_newton_directionwith_lower_bounds;

// --- lower-crate imports -------------------------------------------------
pub(crate) use faer::Side;
pub(crate) use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_atb, fast_av};
pub(crate) use gam_linalg::matrix::{
    DesignMatrix, LinearOperator, SignedWeightsView, SymmetricMatrix,
};
pub(crate) use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, s};
pub(crate) use std::any::Any;
pub(crate) use std::cell::RefCell;
pub(crate) use std::collections::BTreeMap;
pub(crate) use std::sync::atomic::{AtomicUsize, Ordering};
pub(crate) use std::sync::{Arc, Mutex, OnceLock};

// --- descended carriers + the `CustomFamily` trait + fit options + ψ design --
// `pub use gam_problem::*` brings the block data model (`ParameterBlockSpec`,
// `ParameterBlockState`, working sets, channel-Hessians), `CustomFamilyError`,
// `BlockRole`, `RhoPrior`, `EvalMode`, `PseudoLogdetMode`,
// `validate_blockspec_consistency`, and the joint-penalty / psi-terms carriers.
// `pub use gam_model_api::families::custom_family::*` brings `CustomFamily`,
// `BlockwiseFitOptions`, `block_offsets_from_specs`, the outer-derivative
// policy/cost types, and the ψ design-derivative operators.
pub(crate) use gam_model_api::families::custom_family::*;
pub(crate) use gam_problem::*;

// #1521 carve: targeted PUBLIC re-exports so the lifted `fit_orchestration`
// drivers (in gam-models) reach these carriers as `gam_custom_family::<X>`
// (the `pub(crate)` globs above keep them crate-internal).
// Explicit named re-exports shadow the broad `*` globs above (glob imports have
// lower precedence) and — because every name below resolves to the SAME item in
// gam-models' own `gam_model_api`/`gam_problem` globs — they do not introduce an
// E0659 ambiguity in the gam-models facade.
pub use gam_model_api::families::custom_family::{
    BlockwiseFitOptions, CustomFamily, FamilyEvaluation, cost_gated_first_order_max_iter,
    exact_newton_outer_geometry_supports_second_order_solver,
};
pub use gam_problem::{
    AdditiveBlockJacobian, BlockEffectiveJacobian, BlockGeometryDirectionalDerivative,
    BlockWorkingSet, CustomFamilyBlockPsiDerivative, CustomFamilyPsiDerivativeOperator,
    ExactNewtonJointPsiTerms, ExactNewtonOuterObjective, FamilyLinearizationState,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
};

// `whitened_spectrum` is a submodule hosted inside `joint_newton`; re-export it
// at the carrier scope so the trust-region tests reach it through `super::*`.
pub(crate) use joint_newton::whitened_spectrum;

// #1521 carve: the persistent (on-disk) warm-start cache descended into this
// crate alongside the carrier (was crate-root `custom_family_persistent_warm_start`
// in gam-solve). Its three entry points are re-imported into the prelude above
// via `crate::custom_family_persistent_warm_start::{...}`.
mod custom_family_persistent_warm_start;

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
mod psi_design;
mod psi_hyper;
mod warm_start;

// `pub use ...::*` preserves each item's own visibility (pub stays pub,
// pub(crate) stays pub(crate)) so the prior flat-namespace API is unchanged.
pub use assembly::*;
pub use self::block_spec::{
    CoefficientBlockSelector, CoefficientGroupSpec, CoefficientLabel, RealizedCoefficientGroup,
    RealizedCoefficientGroupSpecs, coefficient_label,
};
pub(crate) use self::block_spec::custom_family_block_role;
pub use self::block_spec::validate_blockspecs;
pub(crate) use blockwise_solve::*;
pub use coefficient_groups::*;
pub(crate) use covariance::*;
// Two covariance helpers are part of the public flat-namespace API consumed by
// the relocated families (`crate::{use_joint_matrix_free_path,
// projected_linear_constraint_stationarity_vector}`); surface them publicly
// (the `pub(crate) use covariance::*` glob above keeps them crate-internal).
pub use covariance::{
    JOINT_MATRIX_FREE_MIN_DIM, joint_exact_analytic_outer_hessian_available,
    projected_linear_constraint_stationarity_vector, use_joint_matrix_free_path,
};
pub use fit::*;
pub(crate) use inner_blockwise_fit::*;
pub(crate) use jeffreys::*;
pub(crate) use joint_derivatives::*;
pub use joint_newton::*;
pub(crate) use outer_objective::*;
pub(crate) use penalty_labels::*;
// ψ design-derivative operators / actions / joint-ψ operator / resolvers
// (relocated from the pre-carve monolith, #1521). `pub use ...::*` preserves
// each item's visibility so the 18 `pub` symbols surface as
// `gam_custom_family::*` (and thence `crate::custom_family::*` in
// gam-models via the facade glob).
pub use self::psi_design::*;
pub use psi_hyper::*;
pub use warm_start::*;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
