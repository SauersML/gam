//! Integration-test harness for gam-cli: every module here was a
//! standalone tests/*.rs crate and therefore its own link of gam-cli and
//! its dependency tree. One binary, same tests, same names.

mod bug_hunt_all_zero_count_response_2255;
mod bug_hunt_compact_fit_result_drops_outer_certificate;
mod bug_hunt_expectile_cli_fit_aborts_on_frailty_guard;
mod bug_hunt_explicit_family_emits_wrong_inferred_family_note;
mod bug_hunt_predict_uncertainty_shifts_point_mean_for_curved_link;
mod bug_hunt_sas_link_finalize_inner_cap_leak;
mod bug_hunt_sas_link_outer_inner_cap_guard;
mod frontend_payload_parity_2470;
mod regression_bspline_nonzero_anchor_pin_2297;
mod regression_predict_cli_surfaces_covariance_provenance;
mod regression_predict_uncertainty_point_mean_linear_link;
mod regression_separated_binomial_predict_round_trips_2273;
mod spline_scan_persistence_acceptance_2302;
