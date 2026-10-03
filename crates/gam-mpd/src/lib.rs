//! Program decomposition of a network's parameters (#2951).
//!
//! The object is an executable decomposition of a network's parameterized computation, not a
//! reconstruction of its activations. The engine (`engine`, over `operator_program`,
//! `operator_rewrites`, `factors`, `refit`, `derivatives`) takes the model, imported as an
//! operator program (`import`, `safetensors`), and searches the proposals of its primitive
//! library for a shorter two-part code: program bits (`codec`, `precision`) plus
//! `Σ KL(model ‖ program)/ln 2` over the contract's family (`contract`). A proposal is kept when
//! the measured total drops. The search stops when no proposal is accepted or the budget is spent:
//! the result has no further accepted move under this proposal set, and nothing more is claimed
//! about it. `view` and `printer` print a program's operators and rules.
//!
//! The masked decomposition (`pieces`, `masked`, `blocks`) fits per-input rank-one subcomponents
//! of chosen linear maps through the model's own masked forward. Its mask search is heuristic:
//! derivatives propose flips, and a flip is kept when the measured total drops.
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
//! caller picked the status its computation supports. Roundoff bounds (`bounds`, `verify`,
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

// The kinds of structural proposal the engine searches over.
pub mod fit;

// Operator programs: typed operators with exact interfaces, banded batch execution, message code.
pub mod operator_program;

// Tests of operator programs: code round trips, bands against double-double, exact basis rewrites.
#[cfg(test)]
mod operator_program_tests;

// The declared contract of a program decomposition: load, complete and sampled families.
pub mod contract;

// Program decomposition by two-part code: primitives propose, a change is kept when the measured total drops.
pub mod engine;

// The engine on planted programs: recovery, labelling up to automorphism, restatement invariance.
#[cfg(test)]
mod engine_tests;

// The shared-factor fit on a planted layer: one rule per plane, found without naming the planes.
#[cfg(test)]
mod factors_tests;

// Counterexample-guided refinement: input-space ascent of KL(model ‖ program) feeding the data.
pub mod cegar;

// The refinement verifier on programs with every node kind: gradients, ascent, termination.
#[cfg(test)]
mod cegar_tests;

// The operator-level view of a program: reads, applying node kinds, writes, uses, bits by storage.
pub mod view;

// The human-facing reading of an operator program: rules with bindings, bits by storage, per-input KL.
pub mod printer;

// The printer on a planted two-frequency circuit.
#[cfg(test)]
mod printer_tests;

// Exact path decomposition of a traced value: conditioned on the trace, best first, with a certified remainder.
pub mod paths;

// Refitting a program's reals to its contract: exact Newton–CG on the readout's convex KL.
pub mod refit;

// Exact rewrites of operator programs: constant folding, composition, mixes, stacking, a given character basis.
pub mod operator_rewrites;

// Shared writer factors: operators writing one space factored through one library of directions.
pub mod factors;

// Per-input pieces of one linear map: an overcomplete rank-1 library fitted so that each input
// lists few pieces (listing code plus second-order KL).
pub mod pieces;

// Per-input subcomponents trained through the model's own masked forward; heuristic mask search, a flip
// kept when the measured total drops.
pub mod masked;

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

// Exact directional derivatives of operator programs, and precisions derived from curvature.
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

// Exact invariance quotients: a program's canonical representative under its exact gauges, and the bits saved.
pub mod quotient;

// Quotients on planted programs: bit-identical scale gauges, banded shifts, exact savings.
#[cfg(test)]
mod quotient_tests;

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

// Logit-row comparison with forward-error radii, and exhaustive suprema over a finite family.
pub mod verify;

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

#[cfg(test)]
mod gates_tests;
