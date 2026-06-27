// The encoded-dataset layer was hoisted into the `gam-data` foundation crate
// and is re-exported at the crate root as `gam::data`. It used to live here as
// `inference::data`, and a large body of integration tests (plus any external
// consumer) still names `gam::inference::data`. Keep that path valid by
// aliasing the relocated crate so the extraction does not silently drop a
// public module path it inherited.
pub mod util;
pub use gam_data as data;

// `alo` descended into gam-solve (#1521): genuine REML-evidence numerics whose
// deps are all ≤ gam-solve. Re-exported here so `gam::inference::alo` (named by
// gam-predict/gam-inference conformal, model_comparison, the CLI) resolves
// unchanged.
pub use gam_solve::inference::alo;
pub use gam_sae::inference::atom_lens;
pub mod certificate_impls;
pub mod certificates;
pub use gam_sae::inference::checkpoint_dynamics;
pub use gam_problem::diagnostics;
pub use gam_problem::dispersion_cov;
pub mod fisher_rao;
pub mod functionals;
pub use gam_sae::inference::harvest;
pub use gam_terms::inference::higher_order;
pub mod hmc_io;
// `hmc_io` is the post-rename home of the NUTS/HMC engine that integration
// tests and downstream callers still reach as `inference::hmc`. Keep that path
// resolvable alongside the crate-root `gam::hmc` alias.
pub use hmc_io as hmc;
pub use gam_terms::inference::lawley;
pub use gam_sae::inference::layer_transport;
pub mod model_comparison;
// #1521: pg_gate_evidence/pg_moments descended into gam-solve (reached downward
// by gam_sae::structure_harvest); re-exported here so `gam::inference::{
// pg_gate_evidence, pg_moments}` resolves unchanged.
pub use gam_solve::inference::{pg_gate_evidence, pg_moments};
pub mod gpu_polya_gamma;
pub mod polya_gamma;
pub mod polya_gamma_core;
pub mod posterior;
pub use gam_models::inference::{full_conformal, generative, model, model_payload_builders, predict_io};
pub use gam_terms::inference::formula_dsl;
pub mod probability;
pub use gam_sae::inference::probe_runner;
pub mod quadrature;
// `residual_factor` descended into gam-solve (#1521): the structured-residual
// covariance estimator (#974) whose deps are all ≤ gam-solve (`gam_problem::RowMetric`
// + `gam-linalg`). Re-exported here so `gam::inference::residual_factor` (named by
// `tests/identifiability/misc/structured_residual_974.rs`) resolves unchanged.
pub use gam_solve::inference::residual_factor;
pub mod rho_posterior;
pub use gam_sae::inference::riesz;
pub use gam_solve::row_sampling_measure as row_measure;
pub mod row_metric;
pub mod sample;
pub mod skovgaard;
pub use gam_terms::inference::smooth_test;
pub use gam_sae::inference::steering;
pub use gam_terms::inference::structure_evidence;
pub mod truncated_gaussian;
