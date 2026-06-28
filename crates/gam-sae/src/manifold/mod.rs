//! SAE-manifold term configuration.
//!
//! This is the formal Methodspace row for the SAE-manifold term:
//!
//! ```text
//! Z_i ~= sum_k a_ik g_k(t_ik),     g_k(t) = Phi_k(t) B_k
//! ```
//!
//! Tier assignment:
//!
//! * beta: [`SaeManifoldAtom::decoder_coefficients`] (`B_k`, one block per atom).
//! * ext-coords: [`SaeAssignment`] (`logits -> a_ik` and per-atom
//!   `LatentCoordValues`). Softmax uses the identifiable reference-logit chart
//!   with `K - 1` free assignment coordinates (`0` for `K = 1`). Per-row latent coordinates are written `t`; existing
//!   kernel-shape state remains with carriers such as `SpatialLogKappaCoords`.
//! * rho: [`SaeManifoldRho`] (`lambda_sparse`, `lambda_smooth`, `alpha_kj`) plus
//!   the discrete `K` selected by the Python `compare_models` wrapper.
//!
//! The per-row local block is exactly the audit-revised shape:
//!
//! ```text
//! ext_i = (assignment chart_i, t_i0[0..d_0], ..., t_iK[0..d_K])
//! dim(ext_i) = assignment_dim + sum_k d_k
//! ```
//!
//! [`SaeManifoldTerm::assemble_arrow_schur`] materializes the Gauss-Newton
//! bordered Hessian in that layout and hands it to
//! [`gam_solve::arrow_schur::ArrowSchurSystem`].
//!
//! # Module organization
//!
//! This term is large enough that its concerns live in dedicated submodules,
//! re-exported flat from here so the public surface is unchanged:
//!
//! * [`streaming_plan`] — host/device memory budgeting and the in-core vs
//!   matrix-free streaming admission plan.
//! * [`schedule`] — assignment-temperature annealing and the discrete-`K`
//!   search strategy.
//! * [`atom`] — one manifold atom (`SaeManifoldAtom`): basis topology, the
//!   decoder/frame algebra, the intrinsic arc-length penalty, and the ARD
//!   coordinate prior / Bessel normaliser it rests on.
//! * [`rho`] — the REML-selected continuous hyperparameters and their flat
//!   outer-coordinate layout.
//! * [`kronecker`] — the matrix-free Kronecker-factored β Jacobian primitive.
//! * [`loss`] — the loss breakdown and outer-ρ gradient component value objects.
//! * [`arrow_solver`] — the gauge-deflated arrow-factor solve and the per-row
//!   jet bookkeeping the curvature assembly threads through it.
//! * [`row_layout`] — the per-row active-set layout for sparse assignment.
//! * [`shape_uncertainty`] — the posterior shape-band payload types.
//! * [`certificate`] — the curved-dictionary global-optimality certificate and
//!   the post-fit diagnostics it feeds.
//! * [`term`] — the `SaeManifoldTerm` aggregate, its shared numeric constants,
//!   and the mutable-state snapshot the inner line search restores.
//! * [`construction`] — term construction, accessors, frame/border bookkeeping,
//!   loss/penalty/criterion evaluation, and the arrow-Schur assembly.
//! * [`penalties`] — the live analytic-penalty curvature contributions.
//! * [`fit_drivers`] — gauge canonicalization, the Newton step, and the joint /
//!   fixed-decoder / streaming fit drivers.
//! * [`outer_objective`] — the generic-engine REML outer objective and the
//!   curvature-homotopy entry walk.

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, s};

use std::sync::Arc;

pub(crate) use gam_solve::arrow_schur::{
    ArrowProximalCorrectionOptions, ArrowRowBlock, ArrowSchurError, ArrowSchurSystem,
    ArrowSolveOptions, BetaPenaltyOp, CompositePenaltyOp, DensePenaltyOp, DeviceSaePcgData,
    DeviceSaeSmoothBlock, FactoredFrameGBlock, FactoredFrameKroneckerOp, IbpCrossRowSource,
    IdentityRightKroneckerPenaltyOp, SparseBlockKroneckerPenaltyOp, SparseGBlock,
    StreamingArrowSchur, solve_arrow_newton_step_with_proximal_correction,
    solve_streaming_reduced_beta, solve_with_lm_escalation_inner,
    streaming_cross_row_woodbury_log_det,
};

pub(crate) use gam_terms::analytic_penalties::{
    AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry, DecoderIncoherencePenalty,
    IbpHessianDiagThirdChannels, IsometryPenalty, MechanismSparsityPenalty, NuclearNormPenalty,
    PenaltyTier, PsiSlice, WeightField, resolve_learnable_weight,
};

pub(crate) use gam_terms::latent::{LatentCoordValues, LatentIdMode, LatentManifold};

pub(crate) use crate::criterion_atoms::SaeCriterion;

pub(crate) use crate::certificates::{
    CriterionCertificate, DirectionalSamples, certificate_from_samples,
    deterministic_probe_direction, probe_step,
};

pub(crate) use gam_linalg::faer_ndarray::{
    FaerCholesky, FaerCholeskyFactor, FaerEigh, FaerSvd, fast_ab, fast_abt, fast_atb,
    with_nested_parallel,
};

pub(crate) use gam_linalg::triangular::cholesky_solve_vector;

pub(crate) use gam_solve::arrow_schur::{
    ArrowFactorCache, ArrowRowGaugeDeflation, RowDeflationSpectrum, arrow_factor_max_pivot,
    arrow_factor_min_pivot, solve_arrow_newton_step_with_options,
};

pub(crate) use gam_solve::estimate::EstimationError;

pub(crate) use gam_solve::evidence::arrow_log_det_from_cache;

pub(crate) use gam_solve::rho_optimizer::{
    OuterCapability, OuterEvalOrder, OuterObjective, SeedOutcome,
};
pub(crate) use gam_problem::{DeclaredHessianForm, Derivative, EfsEval, HessianResult, OuterEval};

pub(crate) use gam_solve::structure_search::{CollapseAction, CollapseEvent};

pub(crate) use faer::Side;

// The SAE assignment / basis / frame primitives this term is built from. They
// are re-exported flat here so every submodule reaches them through
// `use super::*` and the public surface is unchanged.
pub use crate::assignment::*;
pub use crate::basis::*;
pub use crate::frames::*;

mod amortized_routing;
mod arrow_solver;
mod atom;
mod certificate;
mod construction;
mod construction_cache_refresh;
mod construction_padded_blocks;
mod construction_reconstruction;
mod fit_drivers;
mod kronecker;
mod loss;
mod outer_objective;
mod pca_seed;
mod penalties;
mod rho;
mod row_layout;
mod schedule;
mod shape_uncertainty;
mod streaming_plan;
mod term;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_parallelism_invariance_1557;

#[cfg(test)]
mod tests_olmo;

#[cfg(test)]
mod tests_row_jet_and_outer_objective_780;

#[cfg(test)]
mod tests_deflation_traces_780;

#[cfg(test)]
mod tests_isometry_exact_hvp_majorizer_457;

#[cfg(test)]
mod sae_contract_probe_tests;

#[cfg(test)]
mod lambda_smooth_1556_tests;

pub use arrow_solver::*;
pub use atom::*;
pub use certificate::*;
pub use construction::*;
pub use construction_cache_refresh::*;
pub use construction_padded_blocks::*;
pub(crate) use kronecker::*;
pub use loss::*;
pub use outer_objective::*;
pub use pca_seed::*;
pub use penalties::*;
pub use rho::*;
pub use row_layout::*;
pub use schedule::*;
pub use shape_uncertainty::*;
pub use streaming_plan::*;
pub use term::*;
