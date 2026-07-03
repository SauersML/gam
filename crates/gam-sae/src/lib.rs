pub mod assignment;
pub mod atom_codes;
pub mod basis;
pub mod candidate_index;
pub mod certificate_impls;
pub mod certificates;
pub mod chart_canonicalization;
pub mod chart_transfer;
pub mod corpus;
pub mod criterion_atoms;
pub mod description_length;
pub mod dictionary_artifact;
pub mod encode;
pub mod frames;
pub mod gpu_kernels;
pub mod hybrid_split;
pub mod identifiability;
pub mod inference;
pub mod k_selection;
pub mod manifold;
pub mod row_jet_program;
pub mod sparse_dict;
pub mod structure_harvest;

// The pre-split engine referenced GPU infrastructure as `crate::gpu::*`; after
// the #1521 split that code lives in the `gam-gpu` crate. Alias it back so the
// `crate::gpu::{device_runtime, pool, linalg_dispatch, ...}` call sites in
// `manifold/` and `gpu_kernels/` resolve unchanged (same shim the top-level
// `gam` crate uses).
pub use gam_gpu as gpu;
