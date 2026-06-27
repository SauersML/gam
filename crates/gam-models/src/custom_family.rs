//! Custom-family namespace shim — reconstructs the pre-carve `crate::custom_family`
//! (== `crate::families::custom_family`) flat namespace from the three crates the
//! original `families/custom_family/` module was split across by the #1521 carve:
//!
//! - **`gam-model-api`** — the `CustomFamily` trait, its evaluation carriers,
//!   fit *options*, and the ψ design-derivative operators
//!   (`gam_model_api::families::custom_family::{family_trait, options, psi_design,
//!   joint_newton_defaults}`).
//! - **`gam-problem`** — the neutral blockwise data model + error/penalty carriers
//!   (`ParameterBlockSpec`, `ParameterBlockState`, `CustomFamilyError`, …) that the
//!   trait layer `use`s internally but does not itself re-export.
//! - **`gam-solve`** — the inner block-coordinate solve, joint (cross-block) Newton,
//!   outer-objective driver, Jeffreys term, covariance/geometry, and the public
//!   `fit_custom_family_with_rho_prior` entry points.
//!
//! This module is a stable re-export so the ~300 `crate::custom_family::*` call
//! sites across the relocated families resolve unchanged.

// Trait layer + options + ψ design (and the gam-problem carriers these re-export).
pub use gam_model_api::families::custom_family::*;

// Neutral blockwise data-model / error carriers that `gam-model-api` consumes
// privately and does not re-export — surface them in the flat namespace.
pub use gam_problem::{CustomFamilyError, ParameterBlockSpec, ParameterBlockState};

// Block-Jacobian / working-set / family-linearization / penalty carriers that
// descended to `gam-problem` (#1521) and are *not* surfaced by either crate's
// `custom_family` glob above. Named re-exports here keep the prior
// `crate::custom_family::{AdditiveBlockJacobian, …}` paths resolving.
pub use gam_problem::{
    AdditiveBlockJacobian, BlockEffectiveJacobian, BlockWorkingSet, FamilyChannelHessian,
    FamilyLinearizationState, PenaltyMatrix, RowScaledJacobian,
};

// Pseudo-log-det mode selector for the custom-family Jeffreys/pseudo-logdet path;
// descended to `gam-problem` (#1521) and surfaced fully-public at
// `gam_problem::PseudoLogdetMode`. Keeps `crate::custom_family::PseudoLogdetMode`
// resolving for the relocated families.
pub use gam_problem::PseudoLogdetMode;

// Solver half (inner blockwise solve, joint Newton, outer objective, Jeffreys,
// covariance, the `fit_custom_family_with_rho_prior` drivers, JOINT_MATRIX_FREE_MIN_DIM,
// …). Extracted (#1521) into the dedicated `gam-custom-family` crate, which sits
// ABOVE gam-solve and below gam-models. This glob re-export reconstructs the prior
// flat namespace so the relocated families resolve `crate::custom_family::*`
// unchanged.
pub use gam_custom_family::*;

// Block-spec validator used by `survival/marginal_slope/fit_entry.rs`; `pub` in
// `gam-custom-family` (`block_spec::validate_blockspecs`).
pub use gam_custom_family::validate_blockspecs;
