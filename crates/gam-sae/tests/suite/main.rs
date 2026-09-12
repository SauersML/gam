//! Integration-test harness for gam-sae: every module here was a
//! standalone tests/*.rs crate and therefore its own link of gam-sae and
//! its dependency tree. One binary, same tests, same names.

mod calendar_ground_truth_benchmark;
mod chart_gluing_1890;
mod chart_gluing_1890_e2e;
mod installed_kkt_scaling_2548;
mod repro_2512;
mod repro_2535;
mod sae_ev_vs_k_1026;
mod saebench_metrics;
mod sparse_lane_no_dense_assignment;
