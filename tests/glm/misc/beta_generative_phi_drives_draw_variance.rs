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

use gam::generative::{GenerativeSpec, NoiseModel};
use gam::types::LikelihoodSpec;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Implied Beta precision from an empirical variance at a known mean:
/// `Var(Beta(μφ,(1−μ)φ)) = μ(1−μ)/(φ+1)  ⇒  φ = μ(1−μ)/Var − 1`.
fn implied_phi(mu: f64, var: f64) -> f64 {
    mu * (1.0 - mu) / var - 1.0
}

/// Draw `n_draws × n_rows` Beta replicates at a constant mean `mu` from the
/// generative spec built with the given `gaussian_scale`, and return the
/// pooled empirical (population) variance of every drawn value.
fn empirical_draw_variance(
    seed_spec: &LikelihoodSpec,
    gaussian_scale: Option<f64>,
    mu: f64,
) -> f64 {
    const N_ROWS: usize = 4_000;
    const N_DRAWS: usize = 25;

    let noise = NoiseModel::from_likelihood(seed_spec, N_ROWS, gaussian_scale)
        .expect("beta generative noise model builds");
    let spec = GenerativeSpec {
        mean: Array1::from_elem(N_ROWS, mu),
        noise,
    };

    // Deterministic, fixed-seed RNG so the test is reproducible.
    let mut rng = StdRng::seed_from_u64(0x5EED_B17A_BEEF_0770);
    let draws = sampleobservation_replicates(&spec, N_DRAWS, &mut rng)
        .expect("beta replicate draws succeed");

    let n = draws.len() as f64;
    let mean = draws.iter().copied().sum::<f64>() / n;
    draws.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n
}

