//! Program decomposition of a network's parameters (#2951).
//!
//! A model is imported as an operator program (`import`, `safetensors`, `operator_program`) and
//! executed on a device (`device_program`, `artifact_device`). The explanation is a library of
//! small learned functions per block (`library_mdl`), fitted end to end by one bits-back code
//! length over interchange experiments of causal abstraction (`interchange`).
//!
//! An explanation is held as an [`artifact::Artifact`]: an executable program bound to the model's
//! places, sent in one message (`codec`, `precision`). Derivatives are analytic (`derivatives`);
//! finite differences belong in tests only.

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


// The site nodes of an imported language model.
pub mod run_check;

// Device fitters of rules and replacements through the model's own forward.
pub mod resident_causal_fit;
// The explanation as a library of learned functions, fitted end to end by variational MDL.
pub mod library_mdl;
pub mod library_readout;
// The graph oracle's checker: a program's nodes and edges executed against M under verbatim experiments.
pub mod graph;
#[cfg(test)]
mod graph_tests;
#[cfg(test)]
mod graph_sites_tests;
// Shared functions of the library: one query-key function read by heads of several layers.
pub mod library_sharing;
// A learned mixture prior over the gate directions: read–write ties found by gradient.
pub mod library_mixture;
// Reusable rule bodies: regions rewritten as calls of learned bodies, bodies shared across sites.
pub mod library_bodies;
// Bodies that read through a head: computations across the attention/MLP boundary.
pub mod library_crossing;
// A removal's least-squares compensation: an MLP's surviving functions take over deleted ones' output.
pub mod library_compensation;
pub mod library_complexity;
// The removal search: groups without effect first, then units ranked by their predicted change of F.
pub mod library_removal;
// Transcoder features as the library's MLP functions: relu(g·x + c) u per feature that fires.
pub mod library_transcoder;
// Native weight edits compiled into M and an explanation through its owners: D(E_e(P)) = e(D(P)).
pub mod weight_edit;
// Adversarial verbatim edits: ascent of the gap over an edit's own parameters.
pub mod adversary;
/// VPD's slices with intrinsic gates as a library explanation.
pub mod library_vpd;
/// Frame starts for library_vpd: M's maps cut exactly through an overcomplete frame per read space.
pub mod library_frame;
// Its posterior resident on the device: sample, Adam step and group divergences without transfers.
pub mod device_posterior;

// Rules of attention heads: one body read through what the decoder holds (a match through an
// earlier head's output-value circuit, a copy through the norm gains), bound per head by a scale.
pub mod rules;

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

// Interchange experiments of causal abstraction on the device: cuts, read and complement patches.
pub mod interchange;
pub mod explanation_battery;
// VPD's MLP subcomponents as parts that edits act on.
pub mod vpd_parts;
/// VPD's slices with intrinsic gates: the start of the main line on vpd4l.
pub mod vpd_start;
#[cfg(test)]
mod interchange_tests;

// Measured interventions on native models for an investigator (#2951).
pub mod oracle;
#[cfg(test)]
mod oracle_tests;
