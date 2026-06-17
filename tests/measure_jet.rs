//! Grouped integration-test crate root for measure_jet tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "measure_jet/measure_jet_acceptance_battery.rs"]
mod measure_jet_acceptance_battery;
#[path = "measure_jet/measure_jet_bms_accuracy_parity_1041.rs"]
mod measure_jet_bms_accuracy_parity_1041;
#[path = "measure_jet/measure_jet_bms_backend.rs"]
mod measure_jet_bms_backend;
#[path = "measure_jet/measure_jet_formula_fit_recovers_signal_on_nonconvergence.rs"]
mod measure_jet_formula_fit_recovers_signal_on_nonconvergence;
#[path = "measure_jet/measure_jet_formula_fit_robustness_sweep.rs"]
mod measure_jet_formula_fit_robustness_sweep;
#[path = "measure_jet/measure_jet_near_miss_decoupling.rs"]
mod measure_jet_near_miss_decoupling;
#[path = "measure_jet/measure_jet_perf_parity.rs"]
mod measure_jet_perf_parity;
#[path = "measure_jet/measure_jet_scale_smoke.rs"]
mod measure_jet_scale_smoke;
#[path = "measure_jet/measure_jet_web_quality.rs"]
mod measure_jet_web_quality;
