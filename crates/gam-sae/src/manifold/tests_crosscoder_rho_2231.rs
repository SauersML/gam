//! #2231 Inc-B CONTRACT pins: `log_lambda_block` as outer ρ coordinates.
//!
//! These tests pin the unification contract BEFORE the outer wiring lands
//! (TDD; red is the honest signal that the coordinates exist but the criterion
//! does not yet price them — see the Inc-B audit on #2231). Green requires:
//!
//! 1. every eval lane rescales the stacked target's block columns from the
//!    pristine copy at ρ-materialization (`√λ_ℓ`, drift-free), and
//! 2. the criterion carries the block λ-dependence: the scaled-block residual
//!    through the existing data term plus the `−Σ_ℓ (n·p_ℓ/2)·log λ_ℓ`
//!    change-of-variables Jacobian. Under the engine's UNIT-dispersion `#F1`
//!    convention the stationary point is `λ_ℓ = n·p_ℓ/R_ℓ`; the shared-φ̂
//!    PROFILED form in #2231 §2a gives `(R_x/p_x)/(R_ℓ/p_ℓ)` instead — the
//!    convention decision is recorded on the issue, and these pins assert only
//!    the properties BOTH conventions share.
//!
//! The scan below is a TEST oracle over a 1-D grid of candidate λ values —
//! it verifies the criterion's shape; production selection stays REML through
//! the outer engine (no grid search in production).

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_crosscoder_rho_2231;` — its single declaration. Saying so in-file
// makes the test scope a claim the compiler enforces rather than one the
// filename merely implies, which is what puts the fixture helpers below in
// the same scope as the `#[test]` fns they serve.
#![cfg(test)]

