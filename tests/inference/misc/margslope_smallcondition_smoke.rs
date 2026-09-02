//! Smoke test: bernoulli marginal-slope must complete quickly on a small,
//! well-conditioned problem.
//!
//! This is a guard against regressions in the inner-Newton / outer-κ
//! interplay that turn small problems into slow problems. The large-scale
//! reproducers (`tests/large_scale_margslope_repro.rs`,
//! `tests/inference/optimization/margslope_inner_pirls_scaling.rs`) sweep n up
//! to 100k+; they are NOT skipped — `#[ignore]` is a hard build abort here
//! (`build.rs` "#[ignore] test" rule, enforcing SPEC.md's ban on the XFAIL
//! pattern) — so this file is not a stand-in for a disabled sibling. It is the
//! cheap always-on guard: a single n=2000 fit asserting both convergence and a
//! wall-clock budget that is generous for a healthy solver but tight enough to
//! catch a slow-loop regression like the CTN exact-fn rejection cycle that
//! recently cost ≥14h of CI.

