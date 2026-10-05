//! Program decomposition of a network's parameters (#2951).
//!
//! A model is imported as an operator program (`import`, `safetensors`, `operator_program`) and
//! executed on a device (`device_program`, `artifact_device`). The explanation is a library of
//! small learned functions per block (`library_mdl`), fitted end to end by one bits-back code
//! length over interchange experiments of causal abstraction (`interchange`).
//!
//! The program search (`program_structure_search`, `program_learned_dag`, `composed_rule_search`
//! and the region modules) explains a model by an [`artifact::Artifact`]: an executable program of
//! priced rules bound to the model's places, sent in one message (`codec`, `precision`) and priced
//! by `acceptance`. Derivatives are analytic (`derivatives`); finite differences belong in tests
//! only.

// Shared test fixtures.
#[cfg(test)]
mod test_support;

// Attention kernels: the host reference and the device execution of one head and of sibling heads.
mod tiled_attention;
mod device_attention;
mod device_heads;

// Prefix, index and subset codes and the bit strings of the artifact's message.
pub mod codec;

// Operator programs: typed operators with exact interfaces, banded batch execution, message code.
pub mod operator_program;

// Tests of operator programs: code round trips, bands against double-double, exact basis rewrites.
#[cfg(test)]
mod operator_program_tests;

// Progress logging and file digests for the drivers.
pub mod engine;

// The explanation as one artifact: program, block bindings, places, exceptions; its message and
// its blocks grafted onto the native model.
pub mod matrix_rule;
pub mod artifact;
pub mod artifact_device;

// The structural cost C(P) and the local disagreement D_local(P) of an explanation.
pub mod acceptance;

#[cfg(test)]
mod acceptance_tests;

// The site nodes of an imported language model.
pub mod run_check;

// Device fitters of rules and replacements through the model's own forward.
pub mod resident_rule_fit;
pub mod resident_causal_fit;
// The explanation as a library of learned functions, fitted end to end by variational MDL.
pub mod library_mdl;
pub mod library_readout;
// Shared functions of the library: one query-key function read by heads of several layers.
pub mod library_sharing;
// Its posterior resident on the device: sample, Adam step and group divergences without transfers.
pub mod device_posterior;
pub mod composed_rule_search;
pub mod program_regions;
pub mod program_joint_regions;
pub mod program_learned_dag;
pub mod program_expression_search;
pub mod program_structure_search;
pub mod parameter_response_program;

// Rules of attention heads: one body read through what the decoder holds (a match through an
// earlier head's output-value circuit, a copy through the norm gains), bound per head by a scale.
pub mod rules;

#[cfg(test)]
mod rules_tests;

// Exact directional derivatives of operator programs.
pub mod derivatives;

// An operator program executed on a device (CUDA, or the host reference), values resident.
pub mod device_program;

#[cfg(test)]
mod device_program_tests;

// Language-model exports and Hugging Face checkpoints as operator programs.
pub mod import;

// Declared-precision real codes and decode-then-evaluate distortion.
pub mod precision;

// A `.safetensors` checkpoint read into exactly widened binary64 arrays.
pub mod safetensors;

// Explicit native intervention-response bindings, validated against native graph laws.
pub mod native_control;

// Linear coefficient fits of program nodes.
pub mod linear_coefficient_fit;
pub mod program_linear_fit;
pub mod native_local_supervision;

/// Fixed activation and shared-operator controls in ordinary differentiable IR.
pub mod intervention_program;

// Interchange experiments of causal abstraction on the device: cuts, read and complement patches.
pub mod interchange;
#[cfg(test)]
mod interchange_tests;

/// Declared rank-one native down-weight perturbations as a full augmented program.
pub mod down_edit_family;
pub mod native_parameter_edit;

/// Exact native primitive-unary capacity-control initialization.
pub mod native_mlp_initialization;

#[cfg(test)]
mod parameter_response_program_tests;

// Ordinary standalone artifact replay with immutable native codeword reuse.
pub mod canonical_artifact;
