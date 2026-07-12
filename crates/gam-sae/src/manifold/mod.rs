//! SAE-manifold term configuration.
//!
//! This is the formal Methodspace row for the SAE-manifold term:
//!
//! ```text
//! Z_i ~= sum_k a_ik g_k(t_ik),     g_k(t) = Phi_k(t) B_k
//! ```
//!
//! # Superposed Geometry (what this machinery is)
//!
//! Thesis: **superposition ambiguity is a flatness disease, and curvature is the
//! cure.** A dictionary of flat co-firing directions is generically
//! non-identifiable — any `GL(d)` recombination of a co-active linear subspace
//! reconstructs identically — whereas curved atoms are generically rigid (jet
//! transversality: second-order osculation of two generic embeddings is
//! infinite-codimension), so their gauge groupoid collapses to
//! `Diff(M) x Sym(F)`. Circles are the optimizer's equilibrium response to
//! superposition, not curiosities. Four faces of the same moduli-geometric object:
//!
//! * **Curvature is identifiability.** Realized-rank / Marchenko-Pastur per atom
//!   is an *empirical Terracini certificate* (border-block Jacobian rank =
//!   `sum_k (d_k+1)`); the `rank_eff==0` veto is the degenerate-tangent exclusion
//!   (and the null atom's RLCT `1/2`). A *centered* circle's cone is the plane, so
//!   it is measure-level identifiable only through its radial law — the `(kappa-2)^2`
//!   ISA producer (support vs measure are complementary halves). Grounding in
//!   [`crate::identifiability`], [`isa_seed`], [`crate::structure_harvest`].
//! * **Persistence is bits.** Log-persistence is an evidence exchange rate — one
//!   nat of log-barcode-length per active row buys one nat per unit codimension;
//!   the RD gain is *activation-space* bits, orthogonal to behavioral nats.
//!   Grounding in [`crate::description_length`], [`persistence`].
//! * **Binding is transport.** Layers act through a transport groupoid; linear
//!   transport of an elliptical atom is forced to be a phase shift `+-theta+phi`;
//!   the residual gauge obstruction is the atom's linear stabilizer.
//!   Grounding in [`chart_canonicalization`], [`certificate`].
//! * **Symmetry is charge.** The rank charge is a running complexity
//!   `lambda(n) = d(-log Z)/d(log n)`; hard rank, the WBIC soft count, and the
//!   RLCT are three regimes of one object, scaled by the atom's occupancy
//!   `n_eff`. Grounding in [`construction`], [`wbic_audit`].
//!
//! Learnability trichotomy: structure resolves in the strict order existence ->
//! dimension -> topology, and *fidelity cannot buy topology, only occupancy can*
//! (why topology labels are the weakest, last-to-converge signal). The identifiability
//! theorem (uniqueness at `sum_k (d_k+1) <= p-1`) is proven for the complex secant
//! calculus and empirically holds for real d=1/d=2 atoms; the completeness of the
//! frame for trained networks is a conjecture. Each mechanism below carries the
//! matching one-line grounding comment at its definition.
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
    DeviceSaeSmoothBlock, FactoredFrameGBlock, FactoredFrameKroneckerOp,
    IdentityRightKroneckerPenaltyOp, SparseBlockKroneckerPenaltyOp, SparseGBlock,
    SparseRankOnePenaltyOp, StreamingArrowSchur, matrix_free_arrow_inverse_apply,
    matrix_free_arrow_operator_apply, prepare_sae_resident_frame, row_sub_floor_null_directions,
    solve_arrow_newton_step_with_proximal_correction, solve_streaming_reduced_beta,
    solve_with_lm_escalation_inner,
};

pub(crate) use gam_terms::analytic_penalties::{
    AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry, DecoderIncoherencePenalty,
    IsometryPenalty, MechanismSparsityPenalty, NuclearNormPenalty,
    OrderedBetaBernoulliHessianDiagThirdChannels, PenaltyTier, PsiSlice, WeightField,
};
// The FFI seed path resolves learnable α through the exact terminal-ρ schedule
// (`gam::terms::sae::manifold::resolve_learnable_weight`), so this re-export
// must be PUBLIC — pub(crate) here broke every CI test-build shard (E0603).
pub use gam_terms::analytic_penalties::resolve_learnable_weight;

pub use gam_terms::latent::LatentManifold;
pub(crate) use gam_terms::latent::{LatentCoordValues, LatentIdMode};

pub(crate) use crate::criterion_atoms::SaeCriterion;

pub(crate) use gam_linalg::faer_ndarray::{
    FaerCholesky, FaerCholeskyFactor, FaerEigh, FaerSvd, fast_ab, fast_abt, fast_atb,
    with_nested_parallel,
};

pub(crate) use gam_linalg::triangular::cholesky_solve_vector;

pub(crate) use gam_solve::arrow_schur::{
    ArrowFactorCache, ArrowRowGaugeDeflation, RowDeflationSpectrum, arrow_factor_max_pivot,
    arrow_factor_min_pivot, probe_undamped_evidence_row_factors,
    solve_arrow_newton_step_with_options,
};

// #988 memory-matrix-free criterion log-det: the reduced-Schur SLQ entry point
// and its shared tuning constants, used when the dense k×k Schur exceeds budget.
pub(crate) use gam_solve::arrow_schur::{
    SCHUR_SLQ_LOGDET_LANCZOS_STEPS, SCHUR_SLQ_LOGDET_PROBES, SCHUR_SLQ_LOGDET_SEED,
};

// #2080 rational-surrogate evidence lane: the build-once threaded entry that
// swaps the SLQ reduced-Schur log|S| for the desync-safe rational surrogate
// (value + ρ-gradient one functional), plus its per-outer-solve frozen state.
pub(crate) use gam_solve::arrow_schur::{
    SurrogateLaneConfig, SurrogateLaneState, hutchinson_reduced_schur_inverse_trace,
    matrix_free_arrow_evidence_log_det_surrogate,
};

pub(crate) use gam_solve::estimate::EstimationError;

pub(crate) use gam_solve::evidence::arrow_log_det_from_cache;

pub(crate) use gam_problem::{DeclaredHessianForm, Derivative, EfsEval, HessianValue, OuterEval};
pub(crate) use gam_solve::rho_optimizer::{
    OuterCapability, OuterConvergedVia, OuterEvalOrder, OuterObjective, SeedOutcome,
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
mod atom_build;
mod basin_bundle;
mod behavior;
mod behavior_entry;
mod behavior_fit;
mod behavior_isometry;
mod certificate;
mod chart_atlas;
mod checkpoint;
mod construction;
mod construction_ard;
mod construction_arrow_schur_assembly;
mod construction_aux_types;
mod construction_cache_refresh;
mod construction_padded_blocks;
mod construction_reconstruction;
mod coordinate_fidelity;
mod cross_fit;
mod crosscoder_drift;
mod crosscoder_fit;
mod curl;
mod derivative_oracle;
mod dual;
mod evaluator_rebuild;
mod fisher_metric;
mod fit_drivers;
mod fit_entry;
mod fit_seed;
mod gauge;
mod graph_atom;
mod inframe_curved;
mod isa_seed;
mod kronecker;
pub mod lift;
mod loss;
mod minimal_seed;
mod oos_entry;
mod oos_logit_seed;
mod outer_objective;
mod pair_kappa;
mod pair_phase;
mod pca_seed;
mod penalties;
mod persistence;
mod rho;
mod row_layout;
mod sandwich;
mod schedule;
mod seed_routing;
mod shape_uncertainty;
mod stagewise;
mod stagewise_seed;
mod steering;
mod stratum_births;
mod streaming_plan;
mod streaming_seed;
mod term;
mod terracini;
mod transport_law;
mod wbic_audit;
mod wbic_dynamics;
mod weight_frame_catalog;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_basin_bundle_envelope;

#[cfg(test)]
mod tests_chart_evaluator_jets;

#[cfg(test)]
mod tests_collapse_prevention;

#[cfg(test)]
mod tests_collapse_2132;

#[cfg(test)]
mod tests_factored_htbeta;

#[cfg(test)]
mod tests_parallelism_invariance_1557;

#[cfg(test)]
mod tests_olmo;

#[cfg(test)]
#[cfg(test)]
mod tests_startup_validation_1782;

#[cfg(test)]
mod tests_zoo_micro_local;

#[cfg(test)]
mod tests_termination_2235;

#[cfg(test)]
mod tests_steering_e4;

#[cfg(test)]
mod tests_steering_crosscoder_2234;

#[cfg(test)]
mod tests_collateral_e2_2234;

#[cfg(test)]
mod tests_schur_seed_refusal_1782;

#[cfg(test)]
mod tests_streaming_materialize_chunk_1801;

#[cfg(test)]
mod tests_streaming_seed_parity_2134;

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
mod tests_2111_dense_torus_acceptance;

#[cfg(test)]
mod tests_rank_charge_2101;

#[cfg(test)]
mod tests_sure_dispersion_2133;

#[cfg(test)]
mod tests_behavioral_fisher_rung1;

#[cfg(test)]
mod tests_inner_budget_trajectory_2015;

#[cfg(test)]
mod tests_two_tier_2023;

#[cfg(test)]
mod tests_tier0_shared_mean_2023;

#[cfg(test)]
mod tests_tier0_primary_path_2023;

#[cfg(test)]
mod tests_certify_external_2266;

#[cfg(test)]
mod tests_structured_residual_floor;

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
mod tests_graph_atom;

#[cfg(test)]
mod tests_graph_spectral_decode;

#[cfg(test)]
mod tests_cocollapse_disjoint_2027;

#[cfg(test)]
mod tests_cocollapse_reseed_2089;

#[cfg(test)]
#[cfg(test)]
mod tests_outer_quasi_laplace_probe_budget_2080;

#[cfg(test)]
#[cfg(test)]
mod lambda_smooth_1556_tests;

#[cfg(test)]
mod tests_behavior_column_equilibration_2015;
#[cfg(test)]
mod tests_behavior_isometry_2015;
#[cfg(test)]
mod tests_behavior_twoblock_rung2;

#[cfg(test)]
mod tests_crosscoder_multiblock;

#[cfg(test)]
mod tests_behavior_qwen_real;

#[cfg(test)]
mod tests_crosscoder_olmo;

#[cfg(test)]
mod tests_stall_diagnostic_2234;

#[cfg(test)]
mod tests_rho_structural_layout_2253;

#[cfg(test)]
mod tests_crosscoder_rho_2231;

#[cfg(test)]
mod tests_checkpoint_resume_wiring;

#[cfg(test)]
mod tests_transport_law;

#[cfg(test)]
mod tests_crosscoder_block_fd_2231;

#[cfg(test)]
mod tests_crosscoder_drift;

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

#[cfg(test)]
mod tests_quality_behavior_calibration_2015;

#[cfg(test)]
mod tests_quality_amplitude_1939;

#[cfg(test)]
mod tests_quality_scale_quotient_2099;

pub use arrow_solver::*;
pub use atom::*;
pub use basin_bundle::*;
pub use behavior::*;
pub use behavior_entry::*;
pub use behavior_fit::*;
pub use behavior_isometry::*;
pub use certificate::*;
pub use chart_atlas::*;
pub use construction_aux_types::*;
pub use construction_cache_refresh::*;
pub use construction_padded_blocks::*;
pub use construction_reconstruction::reconstruct_persisted_atom_set;
pub use construction_reconstruction::steer_persisted_atom_set;
// #16/#2023 — the shared rank-charge DOF core, exposed so the hybrid-split DEMOTE
// gate prices linear/curved candidates in the SAME currency as the joint REML fit.
pub(crate) use construction::realised_rank_charge_dof;
// Jeffreys barrier routing support: the per-assembly frozen coactivation pairs
// and per-atom effective sample sizes carried on `SaeManifoldTerm`.
pub(crate) use penalties::BarrierCoactivationGate;

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

pub use crate::inference::atlas_nerve::AtlasCoveringSide;
pub use atom_build::*;
pub use coordinate_fidelity::*;
pub use cross_fit::*;
pub use crosscoder_drift::*;
pub use crosscoder_fit::*;
pub use curl::*;
pub use derivative_oracle::*;
pub use evaluator_rebuild::*;
pub use fisher_metric::*;
pub use fit_entry::*;
pub use fit_seed::*;
pub use gauge::*;
pub use graph_atom::*;
pub use inframe_curved::*;
pub use isa_seed::*;
pub(crate) use kronecker::*;
pub use loss::*;
pub use minimal_seed::*;
pub use oos_entry::*;
pub use outer_objective::*;
pub use pair_kappa::*;
pub use pair_phase::*;
pub use pca_seed::*;
pub use penalties::*;
pub use persistence::*;
pub use rho::*;
pub use row_layout::*;
pub use sandwich::*;
pub use schedule::*;
pub use seed_routing::*;
pub use shape_uncertainty::*;
pub use stagewise::*;
pub use stagewise_seed::*;
pub use stratum_births::*;
pub use streaming_plan::*;
pub use term::*;
pub use terracini::*;
pub use transport_law::*;
pub use wbic_audit::*;
pub use wbic_dynamics::*;
pub use weight_frame_catalog::*;
