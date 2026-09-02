//! #1033 mechanism (c): the GLM design-moving ψ-sweep is n-INDEPENDENT.
//!
//! The architectural invariant the issue enforces is: n-dependent work happens
//! ONCE per fit (the sufficient-statistic build); the κ/ψ outer loop manipulates
//! only k×k objects whose per-trial cost is O(D²k²) — independent of n. For the
//! non-Gaussian GLM lane the carrier of that invariant is
//! [`FrozenWeightGramTensor`] (`solver/glm_sufficient_lane.rs`): at the warm β it
//! freezes the working weight `W` and builds the weighted-design Chebyshev-in-ψ
//! tensor once, after which every per-trial accessor — the value Gram `XᵀWX(ψ)`,
//! the RHS `XᵀWz(ψ)`, the gradient pair `(∂G/∂ψ, ∂b/∂ψ)`, and the Fisher Hessian
//! block `(∂²G/∂ψ², ∂²b/∂ψ²)` — is served n-free in k-space.
//!
//! This is the algebraic companion to the wall-clock `perf_kappa_loop_n_scaling`
//! measurement: rather than time a fit (which is noisy and gated behind the
//! iso-κ convergence path), it pins the invariant *exactly*. Replicate the SAME
//! `b` distinct base rows `m` times to form `n = m·b`. The weighted-design Gram
//! and all its ψ-derivatives are additive over rows, so the n-row tensor's
//! accessors equal EXACTLY `m ×` the base-row tensor's accessors at every ψ —
//! hence, after dividing by the replication factor, they are BIT-IDENTICAL as n
//! scales at fixed k. Equivalently: the k×k object the outer trial loop touches
//! does not change shape or content-per-unit-data as n grows; the only thing n
//! buys is a constant scale absorbed by the one-time build. That is exactly the
//! "cost/grad/Hessian identical as n scales at fixed k" acceptance for this lane.

