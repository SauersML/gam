//! Finite-difference verification of the Bernoulli `FamilyChannelHessian`.
//!
//! For each family, we:
//!   1. Pick a random pilot state (primary-state vector u_i) and data.
//!   2. Evaluate the row NLL at (u_i + ε e_a + ε e_b), etc. via the closed-form
//!      kernel to obtain the FD second derivative ∂²ρ/∂u_a∂u_b.
//!   3. Compare against `FamilyChannelHessian::fill_subject(i, ...)`.
//!   4. Assert relative error < 1e-6 for all (a,b) entries.
//!
//! The survival marginal-slope fit evaluates its canonical row program
//! directly and no longer exposes a separate identifiability-only Hessian
//! adapter.

// ── Bernoulli marginal-slope (K=1) ────────────────────────────────────────────

