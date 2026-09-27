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
//! Each object names its owner files in parentheses; the module declarations below
//! record which are present.
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
//!   moments, the zonotope of admissible moments, witness masks, ranked supports,
//!   and bounds.
//! * **Mechanism program** (`program`, `fit`, `codec`, `precision`). A typed graph
//!   of Sum, Compose, native primitives, reads and writes, and calls to shared
//!   bodies, with its interface, native reference, code and validity domain.
//!
//! Exact execution under masks belongs to `rewrite` (MLP component coordinates),
//! `gated_rewrite` (gated activations, norms), `attention` (rotary attention under
//! the source's joint softmax) and `block` (attention layers under component
//! reads). Sufficient computational state over finite native responses belongs to
//! `state`. Implementation gauges detected exactly from native tensors (OV
//! passthroughs, SwiGLU units, norm gains, rotary QK, residual basis) belong to
//! `gauge`; mask-gauge covariance, structured paths and commutator facts to
//! `operators`. Plane-rotation
//! recovery from a frozen matrix belongs to `spectral`, and structured edits from a
//! declared single-cycle row action to `cyclic_action`.
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
//! default. Every other tolerance must be derived (a roundoff bound, an eigengap). Derivatives must be analytic; finite differences belong in
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

// Teacher controls at bounded code and the cancelling pair (test builds only).
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

// Exact canonical gauge forms of a native decoder layer, executed against the native layer.
pub mod canonical;

// Tensor registry and the exact residual anchor.
pub mod lift;

// Mask moments, the admissible zonotope, support function and affine-logit adversary.
pub mod moments;

// Exact module splits of plain GELU/ReLU MLPs under the worst-case replacement contract, with certified eta.
pub mod module_split;

// Global versus use-specific edits and occurrence scopes.
pub mod occurrence;

// Implementation-gauge families detected exactly from native tensors, quotiented out of codes.
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

// Exact finite-change accounting through a program: per-source contributions summing to the direct change.
pub mod accounting;

// Forward-error bands of a traced program execution under dense parameters.
pub mod replay;

// Exhaustive verification of an explanation against the native model over a declared finite family.
pub mod verify;

// Exact two-endpoint finite-change operators: softmax, RMSNorm, bilinear products, gated activations.
pub mod secant;

// Exact initial decomposition from native tensors through rank-revealing reads.
pub mod seed;

// Plane-rotation recovery with derived eigengaps.
pub mod spectral;

// Sufficient-state quotient and realization contracts, and the exact linear quotient.
pub mod state;

// Evidence status and ranked robust supports.
pub mod supports;

// Versioned request and report document shared by pyffi and the CLI.
pub mod surface;
