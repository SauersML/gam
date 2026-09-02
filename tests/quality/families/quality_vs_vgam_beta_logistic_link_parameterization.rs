//! End-to-end OBJECTIVE quality: gam's *beta-logistic* inverse link — the exotic,
//! state-bearing variant of the sinh-arcsinh (SAS) link family that reuses the
//! same `SasLinkState` machine but parameterizes the mean through a regularized
//! incomplete-beta map instead of `asinh`/`tanh`.
//!
//! OBJECTIVE METRICS THIS TEST ASSERTS (none is "gam reproduces a peer tool's
//! fitted output"):
//!
//!   (A) TRUTH RECOVERY — primary quality claim. The binomial response is drawn
//!       from a *known* smooth probability surface `p_true(x1,x2) =
//!       I_{logistic(eta_true)}(a,b)`. gam fits `y ~ s(x_1d) + s(x_2d)` through the
//!       beta-logistic link and we assert the fitted success-probability surface
//!       recovers the truth: `RMSE(p_hat, p_true)` is below a principled bar set by
//!       the binomial sampling floor (the irreducible per-point noise SD of a
//!       single Bernoulli draw, averaged over the surface), not by any reference
//!       fit. The fit must also explain the data better than the constant model:
//!       its mean negative log-likelihood beats the intercept-only baseline.
//!
//!   (B) LINK-MATH CORRECTNESS vs MATHEMATICAL GROUND TRUTH — the beta-logistic
//!       inverse link is, by definition, the regularized incomplete beta function
//!       `mu(eta) = I_{logistic(eta)}(a,b)` with derivative `mu'(eta) =
//!       dbeta(u,a,b)*u(1-u)`. Base-R `pbeta`/`dbeta` are the *exact analytic
//!       definition* of those special functions (TOMS-708), so asserting gam's
//!       link code reproduces them is a correctness-vs-ground-truth claim, NOT a
//!       "same as a peer tool" claim. We KEEP this: gam's `mu` must equal `pbeta`,
//!       gam's analytic `d1` must equal the link-scale density `dbeta*u(1-u)` AND
//!       the finite difference of the CDF (catching a wrong-derivative link bug).
//!
//! VGAM's role: DEMOTED to a parameterization cross-check only. We confirm VGAM's
//! `betabinomial(size=1)` success mass equals the Beta mean `a/(a+b)` so the shape
//! map `a=exp(log_delta-eps)`, `b=exp(log_delta+eps)` is the one VGAM's family
//! uses — but gam's pass/fail never depends on matching a VGAM *fit*. The exact
//! beta special functions are base-R's; VGAM is not the source of truth here.
//! Here the spec's true `delta = 1.2`, `epsilon = 0.15` give `a = 1.2*exp(-0.15)`,
//! `b = 1.2*exp(+0.15)`.
//!
//! Requires R; a missing interpreter or package is a hard test failure, never a
//! silent skip (see `src/test_support/reference.rs`).

// True beta-logistic parameters (spec): natural-scale delta = 1.2, epsilon = 0.15.
// The link's additive shape term is `log_delta = ln(delta)`, so a = exp(log_delta
// - epsilon), b = exp(log_delta + epsilon) = 1.2*exp(-/+0.15) — the same Beta
// shapes VGAM's betabinomial mixes over.

