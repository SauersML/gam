//! Program decomposition of a network's parameters (#2951).
//!
//! A model is imported as an operator program (`import`, `safetensors`, `operator_program`) and
//! explained by an [`artifact::Artifact`]: an executable program of priced rules bound to the
//! model's places, sent in one message (`codec`, `precision`). The acceptance path (`acceptance`,
//! `candidate_frontier`, `run_check`) minimises the artifact's structural cost `C(P)` subject to
//! its local fidelity `D_local(P) ≤ δ` and its run fidelity `D_run(P) ≤ ε`, `D_run` measured
//! under declared native interventions (`counterfactual`, `intervention_program`). Candidates are
//! fitted through the model's own forward on a device (`resident_causal_fit`, `device_program`,
//! `artifact_device`).
//!
//! Exact execution belongs to `gated_rewrite` (gated activations, norms), `attention` (rotary
//! attention under the source's joint softmax), `block` and `apply` (native linear reads with
//! their radii), `joint_operators` (gauge-invariant query/key and value/output operators) and
//! `llama_simple_mlp` (VPD's 4-layer target). The edit compiler (`compile`, over `lift` and
//! `gauge`) turns control settings into native parameter edits or infeasibility witnesses.
//!
//! # Evidence
//!
//! Reported quantities carry an evidence status (`supports`): exact (algebraic, or exhaustive
//! over a stated finite family), a uniform bound over a stated region including numerical error,
//! a statistical estimate with its distribution and standard error, a counterexample, or
//! unresolved. The status type checks that each status is well formed; it does not check that a
//! caller picked the status its computation supports. Roundoff bounds (`bounds`,
//! `secant`) are derived from the operations performed; derivatives are analytic, and finite
//! differences belong in tests only.

// Shared planted-rotation fixtures with derived float-defect bounds.
#[cfg(test)]
mod test_support;

// Factored edits of a native linear use site, and its native read.
pub mod apply;

// Component query-key kernel under the source's joint softmax and causal mask.
pub mod attention;
mod tiled_attention;
pub mod query_transition;
mod device_attention;
mod device_heads;

// Native linear reads and RMSNorm evaluations with forward-error radii.
pub mod block;

// KL and total-variation bounds over logit boxes.
pub mod bounds;

// Prefix, subset and graph codes for the global artifact and local packets.
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
pub mod matrix_rule_enumeration;
pub mod unary_rule_bank;
pub mod artifact;
pub mod artifact_device;
pub mod device_family_run;
pub mod missing_interface_bound;

// One acceptance path: minimise C(P) subject to D_local(P) ≤ δ and D_run(P) ≤ ε, over a frontier.
pub mod acceptance;
pub mod candidate_frontier;

#[cfg(test)]
mod acceptance_tests;

// D_run of a language model's artifact under counterfactual's declared episodes.
pub mod run_check;

// The team's pieces (MLP accounts, rules, decomposition coordinates) as proposals to the one search.
pub mod proposals;
pub mod resident_rule_fit;
pub mod resident_causal_fit;
// The explanation as a library of learned functions, fitted end to end by variational MDL.
pub mod library_mdl;
pub mod composed_rule_search;
pub mod program_regions;
pub mod program_joint_regions;
pub mod program_learned_dag;
pub mod program_expression_search;
pub mod program_structure_search;
pub mod parameter_response_program;
pub mod vector_rule_pilot;
pub mod shared_geometry_pilot;
pub mod shared_geometry_transfer;

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

// Model exports (export.json and raw float64 tensors) as operator programs and contracts.
pub mod import;

// Exact masked rewrites of gated units, norms, biases and residual edges.
pub mod gated_rewrite;

// Factored gauge-invariant joint operators: rotary QK planes, OV per head and group, Grams, equality.
pub mod joint_operators;

// The tensor registry: storage, aliases and use sites.
pub mod lift;

// The linear pass-through gauge GL(r) and the certified operator difference of two settings.
pub mod gauge;

// The known-answer toys (induction, modular addition, residual MLPs) against the owners.
#[cfg(test)]
mod known_answer_toys_tests;

// Declared-precision real codes and decode-then-evaluate distortion.
pub mod precision;

// The native edit compiler: control settings to native parameter edits, or infeasibility witnesses.
pub mod compile;

// A `.safetensors` checkpoint read into exactly widened binary64 arrays.
pub mod safetensors;

// The LlamaSimpleMLP decoder (VPD's Pile target) executed from its checkpoint with forward-error radii.
pub mod llama_simple_mlp;

// Exact two-endpoint finite-change operators: softmax and bilinear products.
pub mod secant;

// Evidence status and ranked robust supports.
pub mod supports;

// An MLP accounted for by explicit rules between amplitudes.
pub mod mlp_account;

#[cfg(test)]
mod mlp_account_tests;

// The native decoder under declared interventions, and a replacement's scores against it.
pub mod counterfactual;

#[cfg(test)]
mod counterfactual_tests;

// Isolated replacement writes followed by the unchanged native downstream program.
pub mod local_kl;

// Native-grounded finite-bank MLP rule compiler (all coefficients and maps priced).
pub mod native_mlp;

// Measured-family necessary affine output-rank floors (proposal diagnostics).
pub mod native_mlp_rank;

// Exact decoded native parameter sharing across compacted candidate graphs.
pub mod decoded_intern;

// Native imported attention interfaces without splitting or renumbering.
pub mod attention_map;

// Opt-in native head-only CUDA logits with unchanged CPU metrics.
pub mod native_readout;

// Checked scalar intervals on fixed finite inputs; no acceptance backend switch.
pub mod fixed_logit_interval;
pub mod fixed_metric_device;

// Optional fixed-logit obstruction when two episodes receive one prediction.
pub mod response_collision;

// Explicit native intervention-response bindings, validated against native graph laws.
pub mod native_control;

/// Full-width affine MLP fitting baseline (proposal diagnostics only).
pub mod affine_mlp;
pub mod linear_coefficient_fit;
pub mod program_linear_fit;
pub mod native_local_supervision;

/// Fixed activation and shared-operator controls in ordinary differentiable IR.
pub mod intervention_program;

/// Declared rank-one native down-weight perturbations as a full augmented program.
pub mod down_edit_family;
pub mod native_parameter_edit;

/// Exact native primitive-unary capacity-control initialization.
pub mod native_mlp_initialization;

#[cfg(test)]
mod parameter_response_program_tests;

// Ordinary standalone artifact replay with immutable native codeword reuse.
pub mod canonical_artifact;
