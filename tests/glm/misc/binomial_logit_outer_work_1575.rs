//! Regression guard for #1575 — binomial/logit REML outer-work blow-up.
//!
//! A plain multi-smooth logistic GAM fit reportedly drove ~150 outer REML cost
//! evaluations regardless of `n`. #1575 originally tried to cure this by
//! loosening the adaptive inner-PIRLS KKT tolerance ceiling from the tight inner
//! tolerance (`pirls_config.convergence_tolerance`, ≈1e-10) to a fixed 1e-6.
//! That loosening was found to be INERT for this fit — tight and loose ceilings
//! produce the IDENTICAL converged answer AND the identical outer-eval count —
//! and it was reverted as dead/misleading code. The #1575 outer-work reduction
//! therefore remains an OPEN perf target requiring a convergence-preserving
//! approach (e.g. inner warm-starting), not a tolerance tweak.
//!
//! This test fits a small 3-smooth binomial/logit REML GAM and asserts:
//!   (a) correctness: the optimizer certifies a genuine REML stationary point —
//!       the fit mints at all (sealed convergence evidence, SPEC 20) AND the
//!       final outer gradient clears the
//!       solver's own SCORE-RELATIVE stationarity bound. This fit is weakly
//!       identified (near-collinear monomial bases), so the REML surface is a
//!       flat valley and the residual gradient floors at O(0.1) on a score of
//!       ~390 — exactly the mgcv-aligned score-relative convergence the solver
//!       documents (see `rho_optimizer::bridges`). An absolute 1e-3 gradient
//!       bound is the WRONG criterion here (1e-3 is the absolute floor weakly
//!       identified coordinates cannot reach); the score-relative check still
//!       rejects a genuinely non-stationary stuck/overfit mode.
//!   (b) outer work: a coarse upper-bound regression guard (< 60 evals) that
//!       trips if the outer work blows back up toward the ~150-eval bug regime.
//! It does NOT depend on R / mgcv.

