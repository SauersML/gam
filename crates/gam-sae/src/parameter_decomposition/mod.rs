//! Manifold parameter decomposition (#2951).
//!
//! The object is an executable decomposition of a network's parameterized
//! computation, not a reconstruction of its activations: `crate::manifold` fits
//! `Z_i ~= sum_k a_ik g_k(t_ik)` and has no source-weight action. Every file here
//! must keep each operation tied to the original tensors, give each approximation
//! an exact native reference, and check fidelity under declared finite
//! interventions rather than only at the all-on point.
//!
//! # Four objects
//!
//! Each object names its owner files in parentheses. They land one at a time; the
//! slot list below records which are present.
//!
//! * **Native lift** (`lift`, `occurrence`, `apply`). A tensor registry: stable
//!   ids, shapes, aliases, use sites with the orientation of the matrix each use
//!   multiplies by, and a
//!   teacher fingerprint. A global edit acts on every use of a stored tensor and a
//!   use-specific edit acts on one occurrence; they are different experiments.
//!   Components enter through the exact residual anchor
//!
//!   ```text
//!   Theta(m) = m_Delta Theta_* + B sum_c (m_c - m_Delta) v_c
//!   ```
//!
//!   which equals `sum_c m_c P_c + m_Delta (Theta_* - sum_c P_c)` with
//!   `P_c = B v_c`, so the residual is carried exactly and never refitted. The
//!   anchor must be applied matrix-free. Algebraic equality is not bitwise
//!   equality, so the all-on setting must execute the original tensors on their
//!   original path.
//! * **Parameter field** (`field`). `Gamma(z) = sum_j phi_j(z) B_j` over GAM's
//!   existing bases, with fixed instances `P_c = w_c Gamma(z_c)` and
//!   `v_c = w_c phi(z_c)`. The labels `z_c` and scales `w_c` do not depend on the
//!   input; input dependence only selects or masks instances.
//! * **Ablation geometry** (`moments`, `adversary`, `supports`, `bounds`). Mask
//!   moments, the zonotope of admissible moments, witness masks, supports as
//!   hitting sets, and bounds.
//! * **Mechanism program** (`program`, `fit`, `codec`, `precision`). A typed graph
//!   of Sum, Compose, native primitives, reads and writes, and calls to shared
//!   bodies, with its interface, native reference, code and validity domain.
//!
//! Exact execution under masks belongs to `rewrite` (MLP component coordinates),
//! `gated_rewrite` (gated activations, norms, biases, residuals) and `attention`
//! (the component query-key kernel under the source's joint softmax). Gauge and
//! operator structure belongs to `operators`, `spectral` and `state`.
//!
//! # Types that are never coerced into one another
//!
//! * Four manifolds: a parameter-family label (fixed per instance), a
//!   computational-state coordinate (per input), an implementation-gauge
//!   coordinate, and a permitted structured-edit coordinate (e.g. a rotation
//!   angle).
//! * A global edit and a use-specific edit.
//! * A mask group (tied controls) and a macro (a packaged subgraph with
//!   independent internal controls).
//! * A quotient contract `E' T = g E` and a realization contract `T D = D' g`.
//!
//! # Evidence and inputs
//!
//! Every reported quantity must carry its evidence status: exact (algebraic, or
//! exhaustive over a stated finite family), a uniform bound over a stated region
//! including numerical error, a statistical estimate with its law and standard
//! error, a counterexample, or unresolved (lower witness, upper bound, gap). A
//! result must never return a stronger status than it proved; a stochastic-mask
//! mean, an observed worst case and a certified bound are three different numbers.
//!
//! The mask domain and the fidelity tolerance are experiment declarations with no
//! default. Every other tolerance must be derived (a roundoff bound, an eigengap, a
//! Lipschitz covering). Derivatives must be analytic; finite differences belong in
//! tests only.

// Shared planted-rotation fixtures with derived float-defect bounds.
#[cfg(test)]
mod test_support;

// Blind re-derivation oracles for P7, P15 and P17 against the landed APIs.
#[cfg(test)]
mod oracle_tests;

// Executed-stage receipts against the native lift.
pub mod receipts;

// Structured-edit coordinates from a declared single-cycle row action: closed-form planes, rotation edits, plane code.
pub mod cyclic_action;

// Planted-rotation teacher controls (test builds only).
#[cfg(test)]
mod teacher_tests;

// Nonlinear separation over the moment zonotope: lower witnesses, derived upper bounds.
pub mod adversary;

// Matrix-free structured edits applied to the current intervened input.
pub mod apply;

// Component query-key kernel under the source's joint softmax and causal mask.
pub mod attention;

// A whole pre-norm transformer block under masks: norm, attention, residual, norm, MLP, residual.
pub mod block;

// Mechanism programs over attention-only layers, replayed bit for bit under component masks.
pub mod block_program;

// KL oscillation bound, whole-set composition containment, conservation conditioning.
pub mod bounds;

// Prefix, subset and graph codes for the global artifact and local packets.
pub mod codec;

// Matrix-valued parameter fields over GAM bases, with anchored pullbacks.
pub mod field;

// Parameter-family labels: share and split of fixed components into affine fields, scored on decoded code.
pub mod families;

// Joint finite-intervention objective and structural proposals.
pub mod fit;

// Exact masked rewrites of gated units, norms, biases and residual edges.
pub mod gated_rewrite;

// Tensor registry and the exact residual anchor.
pub mod lift;

// Mask moments, the admissible zonotope, support function and affine-logit adversary.
pub mod moments;

// Global versus use-specific edits and occurrence scopes.
pub mod occurrence;

// Implementation-gauge families detected from native tensors, quotiented out of codes and intervention sets.
pub mod gauge;

// Gauge-covariant group masks, structured parameter paths, Sum and Compose accounting.
pub mod operators;

// Cross-module adversarial and null controls against the landed modules.
#[cfg(test)]
mod controls_tests;

// Declared-precision real codes and decode-then-evaluate distortion.
pub mod precision;

// The typed mechanism program graph and its versioned serialization.
pub mod program;

// Exact component-coordinate MLP program under masks.
pub mod rewrite;

// The component MLP block as a mechanism program, bound to its own tensors.
pub mod rewrite_program;

// Exact initial decomposition from native tensors through rank-revealing reads.
pub mod seed;

// Plane-rotation and response-projector recovery with derived eigengaps.
pub mod spectral;

// Invariant planes of non-orthogonal operators with Stewart subspace certificates.
pub mod schur;

// Sufficient-state quotient and realization contracts.
pub mod state;

// Finite-intervention response metric and local state coordinates with finite checks.
pub mod response_metric;

// Evidence status, failure hypergraph and robust supports.
pub mod supports;

// Versioned request and report document shared by pyffi and the CLI.
pub mod surface;
