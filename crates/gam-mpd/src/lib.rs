//! Program decomposition of a network's parameters (#2951).
//!
//! The object is an executable decomposition of a network's parameterized computation, not a
//! reconstruction of its activations. One engine (`engine`, over `operator_program`,
//! `operator_rewrites`, `factors`, `refit`, `derivatives`) takes the model, imported as an
//! operator program (`import`, `safetensors`), and an optional behaviour (`behaviors`,
//! `causal_states`), and searches for the program with the shortest two-part code: program
//! bits (`codec`, `precision`) plus `Σ KL(model ‖ program)/ln 2` over the contract's family
//! (`contract`). `view` prints a program's components.
//!
//! Exact execution belongs to `gated_rewrite` (gated activations, norms), `attention` (rotary
//! attention under the source's joint softmax), `block` and `apply` (native linear reads with
//! their radii), `joint_operators` (gauge-invariant query/key and value/output operators) and
//! `llama_simple_mlp` (VPD's 4-layer target). The edit compiler (`compile`, over `lift` and
//! `gauge`) turns control settings into native parameter edits or infeasibility witnesses.
//! `theory` states and proves what a certificate means.
//!
//! # Evidence
//!
//! Every reported quantity carries its evidence status (`supports`): exact (algebraic, or
//! exhaustive over a stated finite family), a uniform bound over a stated region including
//! numerical error, a statistical estimate with its law and standard error, a counterexample,
//! or unresolved. A result never returns a stronger status than it proved. Bounds (`bounds`,
//! `verify`, `secant`) are derived (a roundoff bound, an eigengap), never tuned; derivatives
//! are analytic, and finite differences belong in tests only.

// Shared planted-rotation fixtures with derived float-defect bounds.
#[cfg(test)]
mod test_support;

// Factored edits of a native linear use site, and its native read.
pub mod apply;

// Component query-key kernel under the source's joint softmax and causal mask.
pub mod attention;

// Native linear reads and RMSNorm evaluations with forward-error radii.
pub mod block;

// KL and total-variation bounds over logit boxes.
pub mod bounds;

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

// Contract-driven program decomposition: primitives, batched certified MDL search, certificates.
pub mod engine;

// The engine on planted programs: recovery, labelling up to automorphism, restatement invariance.
#[cfg(test)]
mod engine_tests;

// The shared-factor fit on a planted layer: one rule per plane, found without naming the planes.
#[cfg(test)]
mod factors_tests;

// The component-level view of an operator program: reads, laws, writes, uses, bits, unresolved share.
pub mod view;

// Exact path decomposition of a traced value: conditioned on the trace, best first, with a certified remainder.
pub mod paths;

// Refitting a program's reals to its contract: exact Newton–CG on the readout's convex KL.
pub mod refit;

// Exact rewrites of operator programs: constant folding, composition, mixes, character and plane bases.
pub mod operator_rewrites;

// Shared writer factors: operators writing one space factored through one library of directions.
pub mod factors;

// Per-input pieces of one linear map: an overcomplete rank-1 library fitted so that each input
// lists few pieces (listing code plus second-order KL).
pub mod pieces;

// Per-input pieces trained through the model's own masked forward, with exact selection.
pub mod masked;

#[cfg(test)]
mod masked_tests;

#[cfg(test)]
mod pieces_tests;

// Identifiability: the search's ties classified as gauge, abstraction, redundancy or a distinct
// hypothesis with a natively executed witness.
pub mod identify;

// Exact directional derivatives of operator programs, and precisions derived from curvature.
pub mod derivatives;

// Proposal products (ranking, directions, curvature) on the Apple GPU; acceptances stay float64.
pub mod device;

// Model exports (export.json and raw float64 tensors) as operator programs and contracts.
pub mod import;

// Exact masked rewrites of gated units, norms, biases and residual edges.
pub mod gated_rewrite;

// Factored gauge-invariant joint operators: rotary QK planes, OV per head and group, Grams, equality.
pub mod joint_operators;

// The tensor registry: storage, aliases and use sites.
pub mod lift;

// Residual ReLU MLP stacks over their sources: the exact path rewrite and its two-part code fit.
pub mod mlp_paths;

// Exact module splits of plain GELU/ReLU MLPs under the worst-case replacement contract, with certified eta.
pub mod module_split;

// The linear pass-through gauge GL(r) and the certified operator difference of two settings.
pub mod gauge;

// The planted known-answer toys against observability.
#[cfg(test)]
mod state_toys_tests;

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

// Residual-stream observability: readouts pulled back through declared steps into one Gramian.
pub mod state;

// Causal-state machines of a behaviour: finite classes, counters and retrieve registers under one two-part code.
pub mod causal_states;

// Evidence status and ranked robust supports.
pub mod supports;

// The theory: certificate soundness, invariance, identification, causal abstraction, code sensitivity.
pub mod theory;

// Machine checks of the theory's theorems.
#[cfg(test)]
mod theory_tests;

// Behaviour discovery: the family partitioned into groups, each its own subprogram, by one two-part code.
pub mod behaviors;

#[cfg(test)]
mod behaviors_tests;

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
