//! Grouped integration-test crate root for identifiability tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "identifiability/binary_outcome_bms_identifiability.rs"]
mod binary_outcome_bms_identifiability;
#[path = "identifiability/certificate_ledger_unified.rs"]
mod certificate_ledger_unified;
#[path = "identifiability/channel_aware_identifiability_audit.rs"]
mod channel_aware_identifiability_audit;
#[path = "identifiability/constant_curvature_kappa_coverage_sims.rs"]
mod constant_curvature_kappa_coverage_sims;
#[path = "identifiability/constant_curvature_kappa_inference_e2e.rs"]
mod constant_curvature_kappa_inference_e2e;
#[path = "identifiability/constant_curvature_kappa_outer_gradient_fd.rs"]
mod constant_curvature_kappa_outer_gradient_fd;
#[path = "identifiability/constant_curvature_smooth.rs"]
mod constant_curvature_smooth;
#[path = "identifiability/identifiability_audit_hard_halt_gate.rs"]
mod identifiability_audit_hard_halt_gate;
#[path = "identifiability/identifiability_audit_leverage_thresholds.rs"]
mod identifiability_audit_leverage_thresholds;
#[path = "identifiability/identifiability_audit_score_warp_link_dev.rs"]
mod identifiability_audit_score_warp_link_dev;
#[path = "identifiability/identifiability_audit_skewness_aware.rs"]
mod identifiability_audit_skewness_aware;
#[path = "identifiability/identifiability_nullspace_projector_idempotence_bug.rs"]
mod identifiability_nullspace_projector_idempotence_bug;
#[path = "identifiability/ladder_cert_rate_measure.rs"]
mod ladder_cert_rate_measure;
#[path = "identifiability/structured_residual_974.rs"]
mod structured_residual_974;
#[path = "identifiability/topology_mixture_refinement.rs"]
mod topology_mixture_refinement;
#[path = "identifiability/topology_mixture_rung.rs"]
mod topology_mixture_rung;
#[path = "identifiability/topology_race_calibration.rs"]
mod topology_race_calibration;
#[path = "identifiability/topology_two_verdict_race.rs"]
mod topology_two_verdict_race;
#[path = "identifiability/topology_union_candidates.rs"]
mod topology_union_candidates;
