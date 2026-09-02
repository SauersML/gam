//! Debug finite-difference probes on `ExternalJointHyperEvaluator`.
//!
//! These inherent methods expose the dense effective Hessian, its projected
//! log-determinant, and the converged `(η, weights, c)` state at a fixed `theta`
//! so the analytic ψ-trace formulas can be checked against centered finite
//! differences from regression tests — including the #1601-orphaned
//! design-assembly guards re-homed into the separate gam-models crate, which is
//! why these are `pub` rather than `pub(crate)`. They are compiled
//! unconditionally (the workspace ban-scanner forbids feature gating); a `pub`
//! debug helper the production path never calls is inert by construction.

