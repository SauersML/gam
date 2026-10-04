//! Program decomposition of a network's parameters (#2951).
//!
//! A model is imported as an operator program (`import`, `safetensors`, `operator_program`) and
//! decomposed into per-input rank-one subcomponents of chosen linear maps (`pieces`, `masked`,
//! `blocks`), fitted through the model's own masked forward, on the host or a device
//! (`masked_device`, `device_program`, `device_train`), one site at a time (`site_fit`,
//! `sparse_code`). A subcomponent is priced by the description of its weights (`describe`, `codec`,
//! `precision`) plus the KL its error costs. `explanation` runs the fitted sites as one program, and
//! `counterfactual` scores its predicted response to declared interventions against the model's.
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
mod device_attention;

// Native linear reads and RMSNorm evaluations with forward-error radii.
pub mod block;

// KL and total-variation bounds over logit boxes.
pub mod bounds;

// The box claim's worst case, certified by bound propagation through the masked program.
pub mod certify;
#[cfg(test)]
mod certify_tests;

// Prefix, subset and graph codes for the global artifact and local packets.
pub mod codec;

// Operator programs: typed operators with exact interfaces, banded batch execution, message code.
pub mod operator_program;

// Tests of operator programs: code round trips, bands against double-double, exact basis rewrites.
#[cfg(test)]
mod operator_program_tests;

// The contract of an imported model: its input family and readouts.
pub mod contract;

// Progress logging for the drivers.
pub mod engine;

// Per-input pieces of one linear map: an overcomplete rank-1 library fitted so that each input
// lists few pieces (listing code plus second-order KL).
pub mod pieces;

// Per-input subcomponents trained through the model's own masked forward; heuristic mask search, a flip
// kept when the measured total drops.
pub mod masked;

pub mod explanation;

#[cfg(test)]
mod explanation_tests;

#[cfg(test)]
mod masked_tests;

// Rank-k gated subcomponents: which of a library's columns share one gate, chosen by the code.
pub mod blocks;

#[cfg(test)]
mod blocks_tests;

// Structured descriptions of a block: its readers and writers in decodable charts (harmonic,
// frames), its core by its own structure, priced at the KL its error costs.
pub mod describe;

#[cfg(test)]
mod describe_tests;

// The frozen tail of a decoder language model after a decomposed window, as a masked head.
pub mod tail;

#[cfg(test)]
mod tail_tests;

// Atomic checkpoints of a streaming masked fit, so an interrupted run resumes exactly.
pub mod checkpoint;

#[cfg(test)]
mod checkpoint_tests;

#[cfg(test)]
mod pieces_tests;

// Exact directional derivatives of operator programs.
pub mod derivatives;

// Proposal products (ranking, directions, curvature) on the Apple GPU; acceptances stay float64.
pub mod device;

// An operator program executed on a device (CUDA, or the host reference), values resident.
pub mod device_program;

#[cfg(test)]
mod device_program_tests;

// The masked fit's hot path (forward, KL, mask gradients, Fishers, step products) on a device.
pub mod masked_device;

#[cfg(test)]
mod masked_device_tests;

// A masked program's library trained on a device: every step's sets, KL and box charge on it.
pub mod device_train;

#[cfg(test)]
mod device_train_tests;

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

// Dense float64 decompositions on faer with canonical signs.
pub mod dense;

// A `.safetensors` checkpoint read into exactly widened binary64 arrays.
pub mod safetensors;

// The LlamaSimpleMLP decoder (VPD's Pile target) executed from its checkpoint with forward-error radii.
pub mod llama_simple_mlp;

// Exact two-endpoint finite-change operators: softmax and bilinear products.
pub mod secant;

// Evidence status and ranked robust supports.
pub mod supports;

// Exact normalization of operator programs by equality saturation, gains kept symbolic.
pub mod egraph;

// Equality saturation on planted exact equivalences and symbolic gains.
#[cfg(test)]
mod egraph_tests;

// Rule discovery by anti-unification over the saturated e-graph, bindings priced by the codec.
pub mod antiunify;

// Rule discovery on planted subroutines under orthogonal and linear changes of basis, ResidMLP toys, nulls.
#[cfg(test)]
mod antiunify_tests;

// Concepts: co-firing groups of subcomponents, one latent per word, fitted by one KT code; the
// vocabulary a word's computation is described in.
pub mod concepts;

#[cfg(test)]
mod concepts_tests;

// Gate laws: which subcomponents are on for an input, from a small bits-charged law over the
// model's own amplitudes (fit, code, feature screen, decisions).
pub mod gates;

// Libraries fitted on one site's own inputs to its second-order code.
pub mod site_fit;

// Per-input sparse coding of a site's output by its blocks' real contributions, with certified bounds.
pub mod sparse_code;

#[cfg(test)]
mod sparse_code_tests;

// Counterfactual response: an explanation's predicted response to declared interventions
// against the native model's.
pub mod counterfactual;

#[cfg(test)]
mod counterfactual_tests;

// Known-mechanism toys: counterfactual questions and mechanism checks for an explanation.
pub mod toys;

#[cfg(test)]
mod toys_tests;

#[cfg(test)]
mod site_fit_tests;

#[cfg(test)]
mod gates_tests;
