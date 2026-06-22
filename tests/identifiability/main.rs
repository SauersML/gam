//! Grouped integration-test crate root for identifiability tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod binary_outcome_bms_identifiability;
mod certificate_ledger_unified;
mod channel_aware_identifiability_audit;
mod constant_curvature_kappa_coverage_sims;
mod constant_curvature_kappa_inference_e2e;
mod constant_curvature_kappa_outer_gradient_fd;
mod constant_curvature_smooth;
mod identifiability_audit_hard_halt_gate;
mod identifiability_audit_leverage_thresholds;
mod identifiability_audit_score_warp_link_dev;
mod identifiability_audit_skewness_aware;
mod identifiability_nullspace_projector_idempotence_bug;
mod ladder_cert_rate_measure;
mod structured_residual_974;
mod topology_mixture_refinement;
mod topology_mixture_rung;
mod topology_race_calibration;
mod topology_two_verdict_race;
mod topology_union_candidates;
