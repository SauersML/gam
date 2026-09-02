//! Regression guard for issue #770, attacked from the *draws* angle.
//!
//! The committed bug-hunt test
//! (`bug_hunt_beta_generative_noise_ignores_estimated_phi`) checks the canonical
//! mapping `NoiseModel::from_likelihood` at the struct level: given a fitted
//! precision it must produce `NoiseModel::Beta { phi: fitted }` rather than the
//! seed `phi = 1.0`. That pins the *mapping*, but not the user-visible symptom
//! the issue actually reports — that `gam generate` *draws* Beta responses whose
//! empirical dispersion is ~20× too large because the seed precision leaks
//! through.
//!
//! This test closes that gap end-to-end through the real sampler. It composes
//! the generative `NoiseModel` exactly the way `gam generate` does — taking the
//! dispersion from the `gaussian_scale` argument (what the fit records and the
//! CLI forwards), with the *seed* `phi` left at 1.0 on the embedded
//! `Beta { phi }` spec — then draws a large, deterministically-seeded replicate
//! sample and verifies the **empirical** variance of the draws reflects the
//! supplied precision, not the seed. A `Beta(μφ, (1−μ)φ)` draw has variance
//! `μ(1−μ)/(φ+1)`, so the precision implied by the sample variance,
//! `μ(1−μ)/Var − 1`, must track the forwarded `φ`.
//!
//! Two directions are asserted so the test is a tight two-sided guard:
//!   * forwarding a high precision (`gaussian_scale = Some(40)`) yields draws
//!     whose implied precision is ≈ 40 — the bug pinned this at ≈ 1;
//!   * supplying no fitted dispersion (`None`) falls back to the embedded seed
//!     `φ = 1`, so the *same* spec then produces ~20× the variance.
//! The ratio of the two empirical variances therefore lands near
//! `(40+1)/(1+1) ≈ 20.5`, which is exactly the inflation the issue measured.

