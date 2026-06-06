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

use gam::generative::{GenerativeSpec, NoiseModel, sampleobservation_replicates};
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

#[test]
fn beta_generative_draw_variance_tracks_forwarded_phi_not_seed() {
    let mu = 0.6_f64;
    // Embedded *seed* precision is 1.0 — the un-refreshed construction-time
    // value the bug erroneously sampled with.
    let seed_spec = LikelihoodSpec::beta_logit(1.0);
    let forwarded_phi = 40.0_f64;

    // (a) Forward the fitted precision the way `gam generate` does.
    let var_fitted = empirical_draw_variance(&seed_spec, Some(forwarded_phi), mu);
    let phi_fitted = implied_phi(mu, var_fitted);

    // (b) Same spec, but no fitted dispersion supplied → falls back to the seed.
    let var_seed = empirical_draw_variance(&seed_spec, None, mu);
    let phi_seed = implied_phi(mu, var_seed);

    // The forwarded precision must drive the draws: implied φ ≈ 40, not ≈ 1.
    // 100k samples give a tiny sampling error on the variance, so a wide
    // [30, 50] window is conservative while still excluding the buggy φ ≈ 1.
    assert!(
        (30.0..=50.0).contains(&phi_fitted),
        "forwarded precision did not drive Beta draw variance: implied φ={phi_fitted:.3} \
         (empirical var={var_fitted:.6}); expected ≈ {forwarded_phi}. The seed precision \
         (φ=1) is leaking through — this is issue #770."
    );

    // The seed fallback must reproduce the low-precision (φ ≈ 1) draws, proving
    // the value is genuinely sourced from `gaussian_scale` and not a constant.
    assert!(
        (0.7..=1.4).contains(&phi_seed),
        "seed-fallback Beta draws did not reflect the embedded φ=1: implied φ={phi_seed:.3} \
         (empirical var={var_seed:.6})"
    );

    // End-to-end symptom: forwarding the fit's precision must *shrink* the draw
    // variance by ≈ (φ+1)/(1+1) = 20.5× relative to the seed — the exact
    // over-dispersion the issue reported. Require at least a 10× separation.
    assert!(
        var_seed / var_fitted > 10.0,
        "forwarding the fitted precision did not tighten the Beta draws: \
         seed var={var_seed:.6}, fitted var={var_fitted:.6}, ratio={:.2} (expected ≈ 20.5)",
        var_seed / var_fitted
    );
}
