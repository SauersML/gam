//! Integration-test harness for gam-solve: every module here was a
//! standalone tests/*.rs crate and therefore its own link of gam-solve and
//! its dependency tree. One binary, same tests, same names.

mod bug_hunt_psis_light_tail_nan_khat;
mod inner_fit_core_scaling;
mod issue_1017_resident_frame;
mod penalty_logdet_fixed_geometry_2612;
mod penalty_pseudoinverse_range_classification_2730;
mod probe_2714_log_survival_accuracy;
#[cfg(target_os = "linux")]
mod sae_evidence_matvec_1017;
mod shared_dispersion_deviance_roundoff_2730_tests;
