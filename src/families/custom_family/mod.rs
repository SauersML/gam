//! Custom-family blockwise carrier, organized by concern.
//!
//! The module was previously a set of mechanical `include!` line-fragments
//! (`imports.rs`, `family_trait_and_blocks.rs`, `blockwise_solve.rs`,
//! `joint_newton_outer.rs`, `psi_hyper_and_jeffreys.rs`, `tests.rs`) that
//! text-inlined into one flat namespace. It is now split into real submodules,
//! each a single defensible concern:
//!
//! - [`error`]            — the [`CustomFamilyError`] type.
//! - [`penalty`]          — the [`PenaltyMatrix`] carrier.
//! - [`block_spec`]       — the blockwise data model (specs, groups, working sets).
//! - [`family_trait`]     — the [`CustomFamily`] trait + its evaluation carriers.
//! - [`options`]          — fit options, outer-derivative policy, cost models.
//! - [`psi_design`]       — ψ design-derivative operators and joint-ψ terms.
//! - [`blockwise_solve`]  — the inner block-coordinate solve + numeric kernels.
//! - [`joint_newton`]     — joint (cross-block) Newton + trust region + PCG + KKT.
//! - [`outer_objective`]  — the outer (ρ) objective and inner-fit driver.
//! - [`psi_hyper`]        — ψ `HyperCoord` construction + hyper-objective eval.
//! - [`jeffreys`]         — the Jeffreys-prior contribution to the joint objective.
//! - [`covariance`]       — joint covariance/geometry + stationarity/KKT residuals.
//! - [`fit`]              — the public fit entry points + result assembly.
//! - [`coefficient_groups`], [`persistent_cache`] — pre-existing concern modules.
//!
//! Cross-submodule items are `pub(crate)`; each submodule pulls the shared
//! crate-internal imports below in via `use super::*;`.

pub(crate) use crate::faer_ndarray::FaerEigh;
pub(crate) use crate::faer_ndarray::{FaerCholesky, fast_atb, fast_av};
pub(crate) use crate::matrix::{
    DesignMatrix, EmbeddedColumnBlock, LinearOperator, SignedWeightsView, SymmetricMatrix,
    dense_rowwise_kronecker,
};
pub(crate) use crate::pirls::{LinearInequalityConstraints, solve_newton_directionwith_lower_bounds};
pub(crate) use crate::resource::{DerivativeStorageMode, ResourcePolicy};
pub(crate) use crate::solver::active_set::{
    project_stationarity_residual_on_constraint_cone, solve_quadratic_with_linear_constraints,
};
pub(crate) use crate::solver::estimate::reml::penalty_logdet::PenaltyPseudologdet;
pub(crate) use crate::solver::estimate::reml::unified::{
    BlockCoupledOperator, ContractedPsiSecondOrder, ContractedPsiSecondOrderFn,
    DenseSpectralOperator, DispersionHandling, DriftDerivResult, FixedDriftDerivFn,
    HessianDerivativeProvider, HessianOperator, HyperCoord, HyperCoordDrift, HyperCoordPair,
    HyperOperator, MatrixFreeSpdOperator, PenaltySubspaceTrace, ProjectedKktResidual,
    StochasticTraceState, compute_block_penalty_logdet_derivs, exact_pseudo_logdet,
    positive_eigenvalue_threshold, spectral_epsilon, spectral_regularize,
};
pub(crate) use crate::solver::estimate::{
    EstimationError, FitGeometry, ensure_finite_scalar_estimation, validate_all_finite_estimation,
};
pub(crate) use crate::types::{RidgeDeterminantMode, RidgePolicy};
pub(crate) use faer::Side;
pub(crate) use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, s};
pub(crate) use std::any::Any;
pub(crate) use std::cell::RefCell;
pub(crate) use std::collections::{BTreeMap, HashMap};
pub(crate) use std::ops::Range;
pub(crate) use std::sync::atomic::{AtomicUsize, Ordering};
pub(crate) use std::sync::{Arc, Mutex, OnceLock, Weak};
pub(crate) use thiserror::Error;

pub use crate::solver::estimate::reml::unified::{EvalMode, PseudoLogdetMode};

mod error;
mod penalty;
mod block_spec;
mod family_trait;
mod options;
mod psi_design;
mod blockwise_solve;
mod joint_newton;
mod outer_objective;
mod psi_hyper;
mod jeffreys;
mod covariance;
mod fit;

mod coefficient_groups;
mod persistent_cache;

// `pub use ...::*` preserves each item's own visibility (pub stays pub,
// pub(crate) stays pub(crate)) so the prior flat-namespace API is unchanged.
pub use error::*;
pub use penalty::*;
pub use block_spec::*;
pub use family_trait::*;
pub use options::*;
pub use psi_design::*;
pub use blockwise_solve::*;
pub use joint_newton::*;
pub use outer_objective::*;
pub use psi_hyper::*;
pub use jeffreys::*;
pub use covariance::*;
pub use fit::*;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
