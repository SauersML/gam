//! Outer driver for a single fixed-ρ PIRLS fit.
//!
//! Conceptual home of:
//! - `fit_model_for_fixed_rho` and `fit_model_for_fixed_rho_with_adaptive_kkt` —
//!   the entry points that build the working model, run the inner LM loop, and
//!   assemble the final `PirlsResult`.
//! - The Levenberg–Marquardt outer loop wired around `runworking_model_pirls`
//!   (acceptance test, damping schedule, geodesic acceleration trigger).
//! - The adaptive early-exit predicate driving `AdaptiveKktTolerance`.
//!
//! These functions remain defined in [`super`] (the `pirls` parent module) while
//! the file split is being introduced incrementally. This stub establishes the
//! directory entry; subsequent commits move the bodies here without altering the
//! public API: `crate::solver::pirls::fit_model_for_fixed_rho` resolves the same
//! way both before and after.

pub(super) use super::{
    fit_model_for_fixed_rho, fit_model_for_fixed_rho_with_adaptive_kkt,
};
