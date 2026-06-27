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
    FamilyLinearizationState, PenaltyMatrix,
};

// Pseudo-log-det mode selector for the custom-family Jeffreys/pseudo-logdet path;
// descended to `gam-problem` (#1521) and surfaced fully-public at
// `gam_problem::PseudoLogdetMode`. Keeps `crate::custom_family::PseudoLogdetMode`
// resolving for the relocated families.
pub use gam_problem::PseudoLogdetMode;

// Solver half (inner blockwise solve, joint Newton, outer objective, Jeffreys,
// covariance, the `fit_custom_family_with_rho_prior` drivers, JOINT_MATRIX_FREE_MIN_DIM,
// …). These live in `gam-solve` as the orphan top-level modules (`fit`,
// `blockwise_solve`, `joint_newton`, `outer_objective`, `jeffreys`, `covariance`,
// `psi_hyper`, `inner_blockwise_fit`, `joint_derivatives`, `penalty_labels`,
// `coefficient_groups`, `assembly`, `warm_start`, `sensitivity`) that still need to
// be wired into a `gam_solve::custom_family` aggregator module. Until that lands,
// this import is the single localized blocker for the solver-side symbols.
pub use gam_solve::custom_family::*;

// Block-spec validator used by `survival/marginal_slope/fit_entry.rs`. Lives in
// `gam-solve` (`custom_family::block_spec::validate_blockspecs`) but is currently
// re-exported there as `pub(crate)` (custom_family.rs:108) over a `pub(crate) fn`
// (block_spec.rs:124), so this path only resolves once the visibility agent
// promotes both to `pub`. Listed explicitly so the call site compiles the moment
// that promotion lands.
pub use gam_solve::custom_family::validate_blockspecs;
