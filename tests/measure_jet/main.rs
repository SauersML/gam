//! Grouped integration-test crate root for measure_jet tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod measure_jet_acceptance_battery;
mod measure_jet_bms_accuracy_parity_1041;
mod measure_jet_bms_backend;
mod measure_jet_formula_fit_recovers_signal_on_nonconvergence;
mod measure_jet_formula_fit_robustness_sweep;
mod measure_jet_near_miss_decoupling;
mod measure_jet_perf_parity;
mod measure_jet_scale_smoke;
mod measure_jet_web_quality;
