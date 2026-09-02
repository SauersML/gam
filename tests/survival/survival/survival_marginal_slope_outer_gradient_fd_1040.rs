//! Discriminating diagnostic for issue #1040: does the survival marginal-slope
//! outer REML/LAML loop fail to converge because of an objective↔gradient
//! DESYNC (a bug — analytic gradient disagrees with a finite-difference of the
//! criterion, so the trust region chases a phantom descent direction forever),
//! or because of weak IDENTIFIABILITY (a genuinely flat valley — analytic ≈ FD
//! everywhere but a near-zero outer-Hessian eigenvalue)?
//!
//! The fork is a finite difference of the outer gradient at a fixed θ,
//! component by component. The generic outer runner exposes an opt-in
//! structured record containing its real bounded seed, exact ρ/ψ layout,
//! analytic gradient, finite-difference gradient, and stencil steps. This test
//! drives a small survival-MS fit and consumes that typed evidence directly.
//!
//! The audit runs at θ₀ before the potentially long joint spatial loop. Its own
//! `max_outer_iter` is capped, while prerequisite rho-only profiles retain the
//! production budget required to reach that joint problem.

