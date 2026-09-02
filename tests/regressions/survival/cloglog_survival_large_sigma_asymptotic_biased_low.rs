//! Bug hunt: the large-σ closed-form survival approximation is biased low by
//! several percent in the σ ∈ [8, 15] band, where it is engaged.
//!
//! The survival transform / cloglog inverse-link complement
//!
//!     S(mu, sigma) = E[ exp(-exp(eta)) ],   eta ~ N(mu, sigma^2)
//!
//! is reachable through the public lognormal kernel: `K_{0,1}(mu, sigma) = S`,
//! so `log_kernel_term(ctx, 0, 1.0, mu, sigma).0.exp() == S(mu, sigma)`.
//!
//! For `sigma >= CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN` (= 8) the controlled
//! evaluator routes to `cloglog_large_sigma_transition_approx` /
//! `cloglog_survival_extreme_asymptotic`
//! (`src/inference/quadrature.rs`, ~lines 1281–1373), which uses the two-term
//! "sharp transition" split
//!
//!     S ≈ Phi(-mu/sigma) - exp(mu + sigma^2/2) * Phi(-(mu + sigma^2)/sigma).
//!
//! That split replaces the genuine O(1)-wide transition region of
//! `exp(-exp(eta))` around `eta = 0` with a step function, dropping a positive
//! correction of order `phi(mu/sigma)/sigma`. The result is systematically too
//! LOW by ~2–7% across the σ band where the branch is active — well outside any
//! "exact / controlled" claim (the module comment names exactly these points,
//! e.g. (10, 10), as 256-point-GHQ validation targets).
//!
//! Observed (library vs converged reference quadrature):
//!     (0,  8): 0.45088 vs 0.47189  (-4.5%)
//!     (10,10): 0.13684 vs 0.14704  (-6.9%)
//!     (-1, 9): 0.50018 vs 0.51876  (-3.6%)
//!
//! This test compares S against a converged Simpson reference (the reference
//! matches the library to ~1e-15 at small σ, and converges to <1e-9 between
//! step sizes here). When the asymptotic is corrected (transition term added,
//! or the threshold raised so a higher-accuracy branch handles this band), the
//! library value matches the reference and the test passes unchanged.

