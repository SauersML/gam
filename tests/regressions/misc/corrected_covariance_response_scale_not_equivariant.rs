//! Regression: the smoothing-parameter-corrected Bayesian covariance `Vp`
//! (mgcv's `Vp = Vb + J·V_ρ·Jᵀ`, the matrix `predict(..., interval=...)` turns
//! into prediction standard errors) must be response-scale equivariant.
//!
//! For a Gaussian identity-link GAM, replacing `y` by `c·y` (`c > 0`) is an
//! exact rescaling of the model:
//!   - penalized LS estimate  β̂ → c·β̂,
//!   - REML-optimal λ          unchanged (the REML cost gains only a
//!                              ρ-independent (n/2)·log(c²) offset),
//!   - effective df (EDF)       unchanged,
//!   - dispersion              φ̂ → c²·φ̂.
//! So every second-moment object scales by exactly `c²`. The conditional
//! covariance `Vb = φ̂·H⁻¹` does, and `Vp` — documented to be "on the same
//! dispersion scale as `Vb`" — must scale by the same `c²`.
//!
//! The smoothing correction `J·V_ρ·Jᵀ` is built from `J = dβ̂/dρ` (linear in β̂,
//! so `J → c·J`) and the dispersion-free curvature `V_ρ`, hence it already
//! scales as `c²` — exactly like `Vb`. It must be added to `Vb` directly. The
//! bug (#582) multiplied it by the dispersion φ̂ (≈ c²) a second time, so the
//! correction block scaled as `c⁴`, silently inflating every `predict()`
//! interval for large-magnitude responses (~1500× too wide at c = 1000).
//!
//! This test fits the SAME deterministic dataset at response scales 1 and 1000
//! and asserts, in order:
//!   1. premise  — λ and EDF are equivariant,
//!   2. premise  — `Vb` diagonals scale by exactly `c²`,
//!   3. property — `Vp` diagonals scale by the same `c²` (was `c⁴`).

