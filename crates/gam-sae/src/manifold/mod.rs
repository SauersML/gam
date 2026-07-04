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
//!   and loss/penalty/criterion evaluation.
//! * [`construction_arrow_schur_assembly`] — the Gauss-Newton bordered-Hessian
//!   arrow-Schur assembly and its factored β-penalty curvature helpers.
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

pub(crate) use gam_terms::latent::{LatentCoordValues, LatentIdMode};
pub use gam_terms::latent::LatentManifold;

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

// #988 memory-matrix-free evidence log-det: the reduced-Schur SLQ entry point
// and its shared tuning constants, used when the dense k×k Schur exceeds budget.
pub(crate) use gam_solve::arrow_schur::{
    SCHUR_SLQ_LOGDET_LANCZOS_STEPS, SCHUR_SLQ_LOGDET_PROBES, SCHUR_SLQ_LOGDET_SEED,
    matrix_free_arrow_evidence_log_det,
};

pub(crate) use gam_solve::estimate::EstimationError;

pub(crate) use gam_solve::evidence::arrow_log_det_from_cache;

pub(crate) use gam_problem::{DeclaredHessianForm, Derivative, EfsEval, HessianResult, OuterEval};
pub(crate) use gam_solve::rho_optimizer::{
    OuterCapability, OuterEvalOrder, OuterObjective, SeedOutcome,
};

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
mod behavior;
mod behavior_fit;
mod certificate;
mod construction;
mod construction_ard;
mod construction_arrow_schur_assembly;
mod construction_aux_types;
mod construction_cache_refresh;
mod construction_padded_blocks;
mod construction_reconstruction;
mod coordinate_fidelity;
mod cross_fit;
mod fit_drivers;
mod gauge;
mod inframe_curved;
mod isa_seed;
mod kronecker;
mod loss;
mod outer_objective;
mod pair_kappa;
mod pca_seed;
mod penalties;
mod persistence;
mod rho;
mod row_layout;
mod sandwich;
mod schedule;
mod shape_uncertainty;
mod stagewise;
mod streaming_plan;
mod term;
mod wbic_audit;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_chart_evaluator_jets;

#[cfg(test)]
mod tests_factored_htbeta;

#[cfg(test)]
mod tests_bessel_normaliser_1113;

#[cfg(test)]
mod tests_parallelism_invariance_1557;

#[cfg(test)]
mod tests_olmo;

#[cfg(test)]
mod tests_ibp_capacity_1784;

#[cfg(test)]
mod tests_startup_validation_1782;

#[cfg(test)]
mod tests_schur_seed_refusal_1782;

#[cfg(test)]
mod tests_streaming_materialize_chunk_1801;

#[cfg(test)]
mod tests_recovery_split_780;

#[cfg(test)]
mod tests_unit_speed_inloop_2022;

#[cfg(test)]
mod tests_structured_residual_2021;

#[cfg(test)]
mod tests_encode_whitened_gls_2021;

#[cfg(test)]
mod tests_coatom_sigma_coherence_2021;

#[cfg(test)]
mod tests_2101_birth_locus_probe;

#[cfg(test)]
mod tests_rank_charge_2101;

#[cfg(test)]
mod tests_behavioral_fisher_rung1;

#[cfg(test)]
mod tests_two_tier_2023;

#[cfg(test)]
mod tests_tier0_shared_mean_2023;

#[cfg(test)]
mod tests_streaming_efs_cache_1026;

#[cfg(test)]
mod tests_streaming_outer_gradient_2026;

#[cfg(test)]
mod tests_row_jet_and_outer_objective_780;

#[cfg(test)]
mod tests_deflation_traces_780;

#[cfg(test)]
mod tests_logdet_adjoint_780;

#[cfg(test)]
mod tests_pen_fd_780;

#[cfg(test)]
mod tests_isometry_exact_hvp_majorizer_457;

#[cfg(test)]
mod tests_collapse_bar_reachable_rank_1610;

#[cfg(test)]
mod tests_s1_iteration_zero_guard;

#[cfg(test)]
mod sae_contract_probe_tests;

#[cfg(test)]
mod tests_device_engage_1783;

#[cfg(test)]
mod tests_frame_refresh_alpha_grad;

#[cfg(test)]
mod tests_cocollapse_disjoint_2027;

#[cfg(test)]
mod tests_cocollapse_reseed_2089;

#[cfg(test)]
mod tests_outer_reml_probe_budget_2080;

#[cfg(test)]
mod lambda_smooth_1556_tests;

#[cfg(test)]
mod tests_behavior_twoblock_rung2;

#[cfg(test)]
mod tests_ln_sphere_ambient_f4;

#[cfg(test)]
mod tests_inframe_curved_2130;

#[cfg(test)]
mod tests_topology_persistence_f3;

#[cfg(test)]
mod tests_chart_angle_fidelity_2081;

#[cfg(test)]
mod tests_joint_vs_cascade_2131;

pub use arrow_solver::*;
pub use atom::*;
pub use behavior::*;
pub use behavior_fit::*;
pub use certificate::*;
pub use construction_aux_types::*;
pub use construction_cache_refresh::*;
pub use construction_padded_blocks::*;
// #16/#2023 — the shared rank-charge DOF core, exposed so the hybrid-split DEMOTE
// gate prices linear/curved candidates in the SAME currency as the joint REML fit.
pub(crate) use construction::realised_rank_charge_dof;

/// Public single-currency surface for the realised rank-charge DOF: the SAME
/// `realised_rank_charge_dof` the joint REML PROMOTE gate, the hybrid-split
/// DEMOTE gate, and the streaming block ledger all charge, exposed so external
/// drivers (the Mode-A per-block chart pass, the compose/certify report) price
/// candidates with the EXACT criterion instead of re-deriving the formula —
/// re-derivations drift (a ½-factor mismatch was caught in the first
/// re-implementation attempt, which is precisely why this wrapper exists).
/// `gram` is the candidate's weighted design Gram over its `M` basis columns,
/// `decoder` its `M×p` decoder block, `n_eff` the effective sample mass,
/// `p_out` the output dimension, `dispersion` the reconstruction φ̂ feeding the
/// MP floor. No smoothing-penalty term (matches both gate call sites).
pub fn rank_charge_dof(
    gram: &ndarray::Array2<f64>,
    decoder: &ndarray::Array2<f64>,
    n_eff: f64,
    p_out: f64,
    dispersion: f64,
) -> Result<f64, String> {
    construction::realised_rank_charge_dof(gram, decoder, n_eff, p_out, dispersion, 0.0, None)
}

pub use coordinate_fidelity::*;
pub use cross_fit::*;
pub use gauge::*;
pub use inframe_curved::*;
pub use isa_seed::*;
pub(crate) use kronecker::*;
pub use loss::*;
pub use outer_objective::*;
pub use pair_kappa::*;
pub use pca_seed::*;
pub use penalties::*;
pub use persistence::*;
pub use rho::*;
pub use row_layout::*;
pub use sandwich::*;
pub use schedule::*;
pub use shape_uncertainty::*;
pub use stagewise::*;
pub use streaming_plan::*;
pub use term::*;
pub use wbic_audit::*;
