//! Regression pin for #1051 — the euclidean-1D SAE atom must fit a straight
//! line trivially through the production outer engine.
//!
//! A straight line is the canonical euclidean-1D manifold. Planting one in a
//! `p = 3` ambient with a degree-2 euclidean patch (`basis = [1, t, t²]`) makes
//! the decoder design *rank-deficient*: only the linear column carries signal,
//! so the joint Hessian has a genuine near-null direction in the β (decoder)
//! block, OUTSIDE the closed-form chart gauge orbit. Before the fix this made
//! `outer_gradient_conditioning_error` report "analytic outer gradient
//! undefined (joint Hessian numerically singular)" at the continuation spine,
//! every seed was rejected after ~20 s, the inner solve never converged, and
//! the outer BFGS terminated at the `1e12` infeasible sentinel.
//!
//! This test drives the fit *exactly the way production does* — the generic
//! outer cascade (`OuterProblem::run`) around `SaeManifoldOuterObjective`, the
//! same engine `gam-pyffi`'s `sae_manifold_fit` drives — and asserts the fit
//! (a) returns a finite criterion (NOT the `1e12` sentinel) and (b) recovers
//! the line to high reconstruction R².

