//! Owed-work regression gate for GitHub issue #1448 — Negative-Binomial outer
//! θ↔λ alternation (`src/solver/estimate/optimizer.rs`, the bounded loop around
//! lines 1338–1432).
//!
//! ## The fix
//!
//! With NB `theta` ESTIMATED, the λ-search freezes `theta` at the search value
//! (`frozen_negbin_theta`, #1082) so the REML criterion `F(ρ) = REML(ρ, θ_frozen)`
//! is stationary in ρ; the final accept-fit then ML-refreshes `theta` at the
//! converged η. A SINGLE freeze→refresh leaves the selected ρ optimal only for
//! `θ_frozen`, NOT for the refreshed `θ_final` — so the reported `(ρ*, θ_final)`
//! is only jointly stationary if `theta` happened to barely move.
//!
//! The fix (commit `e21e2ad38`) wraps the ρ-search + accept-fit in a bounded
//! loop: after each refit, if the NB `theta` drifted past
//! `NEGBIN_THETA_JOINT_DRIFT_TOL` (5%), re-freeze the search at `θ_final`, reset
//! the outer seed state, and re-run the ρ search; iterate to the joint `(ρ, θ)`
//! fixed point or a round cap (8). For non-NB / user-fixed-θ fits the loop runs
//! exactly once (the criterion `negbin_theta_is_estimated()` is never met), so
//! those fits are byte-identical to the pre-#1448 single pass.
//!
//! ## What this test asserts — public API, non-vacuous
//!
//! The loop state (`frozen_negbin_theta`, `negbin_alternation_round`,
//! `reset_outer_seed_state`) is PRIVATE to the optimizer, so we assert the
//! observable PROPERTY the loop establishes: JOINT STATIONARITY of the reported
//! `(ρ, θ̂)`. Concretely, on overdispersed counts with a strong, wiggly mean and
//! a rich spline basis fit with ESTIMATED theta:
//!
//!   the smoothing parameters ρ = log λ selected by the estimated-θ fit must
//!   equal the ρ selected by a fit with θ held FIXED at the estimated θ̂.
//!
//! That equality is exactly the fixed-point the alternation drives to: ρ is
//! optimal for the θ̂ the fit reports. The λ-search freezes θ at the value
//! captured from the FIRST converged inner solve, whose η has not yet resolved
//! all the mean wiggle, so `θ_frozen` lands materially below the `θ_final` the
//! accept-fit ML-refreshes at the fully-converged η. Before #1448 the single
//! pass therefore reported a ρ optimal for `θ_frozen`, NOT `θ̂ = θ_final` — and on
//! this fixture those ρ vectors differ by ~0.8 in log λ (verified empirically by
//! disabling the alternation loop). The fixed-at-θ̂ fit is the same λ-search
//! machinery with estimation switched off, so under the fix the two ρ vectors
//! coincide to the outer convergence tolerance (< 1e-3) and the assertion holds;
//! on the pre-fix single-pass code the ~0.8 gap blows the 0.5 band and it fails.
//!
//! The companion ASSERT also checks that θ̂ is a genuine finite overdispersed
//! value (not railed to the Poisson-limit clamp), so the NB θ path is actually
//! engaged and the alternation had a real fixed point to reach.

