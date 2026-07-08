pub mod amortized_encoder;
pub mod assignment;
pub mod front_door;
pub mod attention_kernel;
pub mod atom_codes;
pub mod basis;
pub mod basis_gpu;
pub mod candidate_index;
pub mod certificate_impls;
pub mod certificates;
pub mod chart_canonicalization;
pub mod chart_transfer;
pub mod coactivation_conditionality;
pub mod corpus;
pub mod criterion_atoms;
pub mod description_length;
pub mod dictionary_artifact;
pub mod dual_certificate;
pub mod effect_weight;
pub mod encode;
pub mod frames;
pub mod gpu_kernels;
pub mod hybrid_split;
pub mod identifiability;
pub mod inference;
pub mod k_selection;
pub mod manifold;
pub mod nuisance_atlas;
pub mod null_sampler;
pub mod null_battery;
pub mod routability;
pub mod row_jet_program;
pub mod saebench_metrics;
pub mod sparse_dict;
pub mod spectrometer;
pub mod structure_harvest;
pub mod super_resolution;
pub mod tiered;

// The pre-split engine referenced GPU infrastructure as `crate::gpu::*`; after
// the #1521 split that code lives in the `gam-gpu` crate. Alias it back so the
// `crate::gpu::{device_runtime, pool, linalg_dispatch, ...}` call sites in
// `manifold/` and `gpu_kernels/` resolve unchanged (same shim the top-level
// `gam` crate uses).
pub use gam_gpu as gpu;
