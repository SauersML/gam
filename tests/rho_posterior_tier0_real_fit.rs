//! #938 Tier-0 marginal-smoothing certificate against a REAL fit artifact.
//!
//! The PSIS `ρ`-uncertainty certificate (`src/inference/rho_posterior.rs`) is
//! unit-tested against closed-form Gaussian / heavy-tail fixtures. This test
//! exercises the *objective-lifecycle seam*: a genuine `fit_from_formula` GAM
//! must produce the certificate from its own converged REML objective and
//! surface it on the fit artifact, so the tiers that consume it (1-2) have a
//! real entry point. It asserts the certificate is present and structurally
//! sound, and that it is deterministic across identical fits.

use csv::StringRecord;
use gam::inference::data::EncodedDataset;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

const N: usize = 300;
const SIGMA: f64 = 0.30;

/// Known smooth-plus-linear ground truth, the canonical `s(x) + z` design that
/// genuinely exercises the smoothing-parameter (ρ) machinery the certificate
/// is about.
fn mu_true(x: f64, z: f64) -> f64 {
    (2.0 * PI * x).sin() + 0.6 * z
}

fn build_dataset(seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|_| {
            let x = unif.sample(&mut rng);
            let z = unif.sample(&mut rng);
            let y = mu_true(x, z) + noise.sample(&mut rng);
            StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn fit_and_take_certificate(
    seed: u64,
) -> (f64, gam::inference::rho_posterior::RhoPosteriorCertificate) {
    let ds = build_dataset(seed);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x) + z", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    let reml_score = fit.fit.reml_score;
    let cert = fit.fit.artifacts.rho_posterior_certificate.clone().expect(
        "a smooth-term Gaussian GAM has ρ parameters and an SPD outer Hessian, so the \
             Tier-0 ρ-posterior certificate must be present on the real fit artifact",
    );
    (reml_score, cert)
}

/// The seam delivers: a real fit produces a structurally valid Tier-0
/// certificate, and the importance weights it carries are a proper
/// self-normalized distribution with a sane effective sample size.
#[test]
fn real_gaussian_fit_carries_a_sound_tier0_certificate() {
    init_parallelism();
    let (_reml_score, cert) = fit_and_take_certificate(938_001);

    assert!(
        cert.k_hat.is_finite(),
        "the Pareto tail shape k̂ must be finite, got {}",
        cert.k_hat
    );
    assert!(cert.n_samples >= 2, "the certificate must draw proposals");

    // Self-normalized weights: non-negative, finite, summing to 1.
    let sum: f64 = cert.weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "importance weights must self-normalize to 1, got {sum}"
    );
    assert!(
        cert.weights.iter().all(|&w| w.is_finite() && w >= 0.0),
        "importance weights must be finite and non-negative"
    );

    // Kish ESS lies in (0, M] and is consistent with the carried weights.
    assert!(
        cert.effective_sample_size > 0.0
            && cert.effective_sample_size <= cert.n_samples as f64 + 1e-6,
        "ESS {} must lie in (0, M={}]",
        cert.effective_sample_size,
        cert.n_samples
    );
    let sum_sq: f64 = cert.weights.iter().map(|&w| w * w).sum();
    let ess_from_weights = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };
    assert!(
        (cert.effective_sample_size - ess_from_weights).abs() < 1e-6,
        "ESS {} must equal 1/Σw² = {}",
        cert.effective_sample_size,
        ess_from_weights
    );
}

/// The certificate is deterministic: the fixed-seed proposal stream means two
/// fits of identical data yield bit-identical `k̂` (the lifecycle seam injects
/// the same live criterion both times).
#[test]
fn tier0_certificate_is_deterministic_across_identical_fits() {
    init_parallelism();
    let (score_a, a) = fit_and_take_certificate(938_002);
    let (score_b, b) = fit_and_take_certificate(938_002);
    assert_eq!(
        score_a.to_bits(),
        score_b.to_bits(),
        "identical fits must reach the same REML score"
    );
    assert_eq!(
        a.k_hat.to_bits(),
        b.k_hat.to_bits(),
        "the fixed-seed certificate must give bit-identical k̂ across identical fits"
    );
    assert_eq!(a.n_samples, b.n_samples);
}
