//! #2728 — the two published coefficient covariances must not disagree by
//! orders of magnitude, and the sigma-point nodes that build the corrected one
//! must sit where the posterior actually has mass.
//!
//! # What went wrong
//!
//! `beta_covariance_corrected()` (`Vp`) is assembled by the sigma-point
//! cubature branch as `φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂(ρ)]`, a two-point quadrature
//! for `ρ ~ N(ρ̂, V_ρ)` with each node one posterior sd out along a ρ-Hessian
//! eigendirection. The step was taken from the QUADRATIC model of the
//! criterion, `σ_j^{-1/2}`, and never checked against the criterion it was
//! sampling.
//!
//! At a SATURATED smoothing direction that check fails catastrophically. By the
//! exact reparameterisation identity `H_ρ = diag(λ)·H_λ·diag(λ) + diag(g_ρ)`,
//! the ρ-curvature at `λ = 7.2e-9` is ~0 because `λ²` multiplies it, not
//! because the profile is flat. So `σ⁻¹ = 9.5e4`, the step was **308 in
//! log-λ**, and the node landed at a criterion **3309 nats** above the optimum
//! — posterior weight `e^-3309` — while carrying weight ½. On the fixture
//! below that inflated the reported SE by 8.1× over the conditional `Vb` and
//! 11.1× over the estimator's own Monte-Carlo sampling spread.
//!
//! # What is asserted here
//!
//! Three independent angles on the same root cause, so a regression cannot slip
//! through by satisfying one of them:
//!
//! 1. **The node placement itself.** Every node the correction was built from
//!    sits at a criterion rise of order `PROFILE_SIGMA_RISE = 1/2`, which is
//!    the level a one-sigma node is *asserted* to occupy. This is exact, needs
//!    no Monte Carlo, and is the defect stated in its own terms: the number was
//!    3309.
//! 2. **The two published objects agree in magnitude.** The cubature
//!    correction and the first-order `J·V_ρ·Jᵀ` estimate the SAME quantity; a
//!    refinement that differs from what it refines by orders of magnitude is
//!    not a refinement. The measured ratio of traces was 9993 before the fix.
//! 3. **Calibration against the truth.** With `X` held fixed and only the
//!    Gaussian noise redrawn, the Monte-Carlo spread of `x'β̂` over refits is
//!    exactly what the covariance claims to be, with no misspecification in the
//!    comparison. `Vp` must be within a bounded factor of it.

// ─── Fixture: the #2728 configuration ───────────────────────────────────────
//
// Hybrid anisotropic Duchon on 4 PC coordinates, `K = 80` farthest-point
// centers, Gaussian identity, `n = 4000`. This is the configuration the issue
// measured, reduced only in the number of Monte-Carlo refits.

