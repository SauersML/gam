//! Grouped integration-test crate root for SAE / manifold-SAE tests (issue #1146).
//!
//! The formerly-top-level `tests/sae_*.rs` crates are included here as modules so
//! they link as ONE binary. Add new sae_* tests as a module here.

#[path = "sae/sae_arrow_schur_large_scale.rs"]
mod sae_arrow_schur_large_scale;
#[path = "sae/sae_atom_smooth_structure_evidence.rs"]
mod sae_atom_smooth_structure_evidence;
#[path = "sae/sae_audit_is_invoked.rs"]
mod sae_audit_is_invoked;
#[path = "sae/sae_curvature_estimand_sims.rs"]
mod sae_curvature_estimand_sims;
#[path = "sae/sae_encode_atlas_certified.rs"]
mod sae_encode_atlas_certified;
#[path = "sae/sae_encode_throughput_bench.rs"]
mod sae_encode_throughput_bench;
#[path = "sae/sae_ev_vs_k_frontier.rs"]
mod sae_ev_vs_k_frontier;
#[path = "sae/sae_exact_orbit_certificate.rs"]
mod sae_exact_orbit_certificate;
#[path = "sae/sae_factored_frame_solve.rs"]
mod sae_factored_frame_solve;
#[path = "sae/sae_functional_metric_identifiability.rs"]
mod sae_functional_metric_identifiability;
#[path = "sae/sae_grassmann_frame_fit_battery.rs"]
mod sae_grassmann_frame_fit_battery;
#[path = "sae/sae_incoherence_phase_diagram.rs"]
mod sae_incoherence_phase_diagram;
#[path = "sae/sae_k1_periodic_p2048_profile.rs"]
mod sae_k1_periodic_p2048_profile;
#[path = "sae/sae_manifold_duchon_affine_inner_solve.rs"]
mod sae_manifold_duchon_affine_inner_solve;
#[path = "sae/sae_manifold_euclidean_k2_terminates.rs"]
mod sae_manifold_euclidean_k2_terminates;
#[path = "sae/sae_manifold_euclidean_line_fit.rs"]
mod sae_manifold_euclidean_line_fit;
#[path = "sae/sae_manifold_gauge_deflated_evidence.rs"]
mod sae_manifold_gauge_deflated_evidence;
#[path = "sae/sae_manifold_joint_two_circle_recovery.rs"]
mod sae_manifold_joint_two_circle_recovery;
#[path = "sae/sae_manifold_k1_circle_radial_bias.rs"]
mod sae_manifold_k1_circle_radial_bias;
#[path = "sae/sae_manifold_k_ladder_recovery.rs"]
mod sae_manifold_k_ladder_recovery;
#[path = "sae/sae_manifold_p_greater_than_d.rs"]
mod sae_manifold_p_greater_than_d;
#[path = "sae/sae_manifold_planted_bifurcation.rs"]
mod sae_manifold_planted_bifurcation;
#[path = "sae/sae_manifold_predict_oos_recovers_from_non_pd_row_block.rs"]
mod sae_manifold_predict_oos_recovers_from_non_pd_row_block;
#[path = "sae/sae_manifold_reconstruction_parity.rs"]
mod sae_manifold_reconstruction_parity;
#[path = "sae/sae_manifold_small_n_circle_seed_accept.rs"]
mod sae_manifold_small_n_circle_seed_accept;
#[path = "sae/sae_manifold_sphere_torus_topologies.rs"]
mod sae_manifold_sphere_torus_topologies;
#[path = "sae/sae_outer_gradient_fd_gate.rs"]
mod sae_outer_gradient_fd_gate;
#[path = "sae/sae_replicate_gauge_agreement.rs"]
mod sae_replicate_gauge_agreement;
#[path = "sae/sae_residual_gauge.rs"]
mod sae_residual_gauge;
#[path = "sae/sae_sphere_chart_canonicalization.rs"]
mod sae_sphere_chart_canonicalization;
#[path = "sae/sae_streaming_arrow_schur_contract.rs"]
mod sae_streaming_arrow_schur_contract;
#[path = "sae/sae_torus_chart_canonicalization.rs"]
mod sae_torus_chart_canonicalization;
#[path = "sae/sae_unit_speed_chart_canonicalization.rs"]
mod sae_unit_speed_chart_canonicalization;
