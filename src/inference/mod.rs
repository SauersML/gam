// The encoded-dataset layer was hoisted into the `gam-data` foundation crate
// and is re-exported at the crate root as `gam::data`. It used to live here as
// `inference::data`, and a large body of integration tests (plus any external
// consumer) still names `gam::inference::data`. Keep that path valid by
// aliasing the relocated crate so the extraction does not silently drop a
// public module path it inherited.
pub use crate::data;

pub mod alo;
pub mod atom_lens;
pub mod certificate_impls;
pub mod certificates;
pub mod checkpoint_dynamics;
pub mod diagnostics;
pub mod dispersion_cov;
pub mod fisher_rao;
pub mod formula_dsl;
pub mod full_conformal;
pub mod functionals;
pub mod generative;
pub mod harvest;
pub mod higher_order;
pub mod hmc_io;
pub mod lawley;
pub mod layer_transport;
pub mod model;
pub mod model_comparison;
pub mod model_payload_builders;
pub mod pg_gate_evidence;
pub mod pg_moments;
pub mod polya_gamma;
pub mod polya_gamma_core;
pub mod posterior;
pub mod predict_io;
pub mod probability;
pub mod probe_runner;
pub mod quadrature;
pub mod residual_factor;
pub mod rho_posterior;
pub mod riesz;
pub mod row_measure;
pub mod row_metric;
pub mod sample;
pub mod skovgaard;
pub mod smooth_test;
pub mod steering;
pub mod structure_evidence;
pub mod truncated_gaussian;
