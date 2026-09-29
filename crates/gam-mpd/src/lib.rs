//! Manifold parameter decomposition (#2951).
//!
//! The object is an executable decomposition of a network's parameterized
//! computation, not a reconstruction of its activations: `gam_sae::manifold` fits
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
//! * **Evidence and bounds** (`supports`, `bounds`). The evidence status of every
//!   reported number, and the bounds it is stated with.
//! * **Code and proposals** (`fit`, `codec`, `precision`). Exact code lengths,
//!   declared-precision real codes, and structural proposals decided on decoded
//!   code.
//!
//! Exact execution under masks belongs to `rewrite` (MLP component coordinates),
//! `gated_rewrite` (gated activations, norms), `attention` (rotary attention under
//! the source's joint softmax) and `block` (attention layers under component
//! reads). Sufficient computational state over finite native responses belongs to
//! `state`. Implementation gauges detected exactly from native tensors (OV
//! passthroughs, SwiGLU units, norm gains, rotary QK, residual basis) belong to
//! `gauge`; mask-gauge covariance, structured paths and commutator facts to
//! `operators`. Plane-rotation
//! recovery from a frozen matrix belongs to `spectral`.
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

// Executed-stage receipts against the native lift.
pub mod receipts;


// Matrix-free structured edits applied to the current intervened input.
pub mod apply;

// Component query-key kernel under the source's joint softmax and causal mask.
pub mod attention;

// A whole pre-norm transformer block under masks: norm, attention, residual, norm, MLP, residual.
pub mod block;

// KL oscillation bound, whole-set composition containment, conservation conditioning.
pub mod bounds;

// Prefix, subset and graph codes for the global artifact and local packets.
pub mod codec;

// Matrix-valued parameter fields over GAM bases, with anchored pullbacks.
pub mod field;

pub mod finite_grid;

// Joint finite-intervention objective and structural proposals.
pub mod fit;

// Exact masked rewrites of gated units, norms, biases and residual edges.
pub mod gated_rewrite;

// Exact canonical gauge forms of a native decoder layer, executed against the native layer.
pub mod canonical;

// Factored gauge-invariant joint operators: rotary QK planes, OV per head and group, Grams, equality.
pub mod joint_operators;

// Tensor registry and the exact residual anchor.
pub mod lift;

// Exact module splits of plain GELU/ReLU MLPs under the worst-case replacement contract, with certified eta.
pub mod module_split;

// Global versus use-specific edits and occurrence scopes.
pub mod occurrence;

// Implementation-gauge families detected exactly from native tensors, quotiented out of codes.
pub mod gauge;

// The function-level fibre bounded from a parameter Jacobian: the oracle a gauge census is checked against.
pub mod fibre;

// The gauge census of a decoder layer and of a tied residual stream.
pub mod gauge_census;

// Gauge-covariant group masks, structured parameter paths, Sum and Compose accounting.
pub mod operators;

// The planted known-answer toys against observability and the linear quotient.
#[cfg(test)]
mod state_toys_tests;

// The planted known-answer toys against exhaustive verification.
#[cfg(test)]
mod verify_toys_tests;

// The planted known-answer toys against plane-rotation recovery.
#[cfg(test)]
mod spectral_toys_tests;

// Declared-precision real codes and decode-then-evaluate distortion.
pub mod precision;

// Exact component-coordinate MLP program under masks.
pub mod rewrite;

// Exhaustive verification of an explanation against the native model over a declared finite family.
pub mod verify;

// The native edit compiler: control settings to native parameter edits, or infeasibility witnesses.
pub mod compile;

// Dense float64 decompositions on faer with canonical signs, for the probes.
pub mod dense;

// Exact two-endpoint finite-change operators: softmax, RMSNorm, bilinear products, gated activations.
pub mod secant;

// Plane-rotation recovery with derived eigengaps.
pub mod spectral;

// The sign-gated split of a SwiGLU block, its certified correction bounds and ReLU replacement contract.
pub mod sign_gated;

// Sufficient-state quotient and realization contracts, and the exact linear quotient.
pub mod state;

// Evidence status and ranked robust supports.
pub mod supports;

// The theory: certificate soundness, invariance, identification, causal abstraction, code sensitivity.
pub mod theory;

// Machine checks of the theory's theorems.
#[cfg(test)]
mod theory_tests;

// Versioned request and report document shared by pyffi and the CLI.
pub mod surface;
