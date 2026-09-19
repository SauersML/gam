//! #938 Tier-0 marginal-smoothing adequacy diagnostic against a REAL fit artifact.
//!
//! The PSIS `ρ`-uncertainty adequacy diagnostic (`src/inference/rho_posterior.rs`) is
//! unit-tested against closed-form Gaussian / heavy-tail fixtures. This test
//! exercises the *objective-lifecycle seam*: a genuine `fit_from_formula` GAM
//! must produce the diagnostic from its own converged REML objective and
//! surface it on the fit artifact, so the tiers that consume it (1-2) have a
//! real entry point. It asserts the diagnostic is present and structurally
//! sound, and that it is deterministic across identical fits.

use csv::StringRecord;
use gam::inference::data::EncodedDataset;
use gam::inference::rho_posterior::RhoPosteriorOutcome;
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
/// genuinely exercises the smoothing-parameter (ρ) machinery the diagnostic
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

fn fit_and_take_adequacy(
    seed: u64,
) -> (f64, gam::inference::rho_posterior::RhoPosteriorAdequacy) {
    let ds = build_dataset(seed);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x) + z", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    let reml_score = fit
        .fit
        .reml_score()
        .expect("the fit reports a REML/LAML criterion");
    let adequacy = match &fit.fit.artifacts.rho_posterior {
        RhoPosteriorOutcome::Assessed(adequacy) => adequacy.clone(),
        other => panic!(
            "a smooth-term Gaussian GAM has ρ parameters and an SPD outer Hessian, so the \
             Tier-0 ρ-posterior seam must grade the real fit artifact, got {other:?}"
        ),
    };
    (reml_score, adequacy)
}

/// The seam delivers: a real fit carries an Assessed Tier-0 outcome with a finite
/// tail shape and a Kish effective sample size inside its bounds.
#[test]
fn real_gaussian_fit_carries_a_sound_tier0_adequacy_diagnostic() {
    init_parallelism();
    let (_reml_score, adequacy) = fit_and_take_adequacy(938_001);

    assert!(
        adequacy.k_hat.is_finite(),
        "the Pareto tail shape k̂ must be finite, got {}",
        adequacy.k_hat
    );
    assert!(adequacy.n_samples >= 2, "the diagnostic must draw proposals");

    // Kish's (Σw)²/Σw² over the M self-normalized weights lies in [1, M]: Σw = 1
    // and Cauchy–Schwarz give 1/M ≤ Σw² ≤ 1. Both edges carry the M-term
    // summations' relative rounding M·ε.
    let m = adequacy.n_samples as f64;
    let rounding = 1.0 + m * f64::EPSILON;
    assert!(
        adequacy.effective_sample_size * rounding >= 1.0
            && adequacy.effective_sample_size <= m * rounding,
        "ESS {} must lie in [1, M = {m}]",
        adequacy.effective_sample_size
    );
}

/// The diagnostic is deterministic: the fixed-seed proposal stream means two
/// fits of identical data yield bit-identical `k̂` (the lifecycle seam injects
/// the same live criterion both times).
#[test]
fn tier0_adequacy_is_deterministic_across_identical_fits() {
    init_parallelism();
    let (score_a, a) = fit_and_take_adequacy(938_002);
    let (score_b, b) = fit_and_take_adequacy(938_002);
    assert_eq!(
        score_a.to_bits(),
        score_b.to_bits(),
        "identical fits must reach the same REML score"
    );
    assert_eq!(
        a.k_hat.to_bits(),
        b.k_hat.to_bits(),
        "the fixed-seed diagnostic must give bit-identical k̂ across identical fits"
    );
    assert_eq!(a.n_samples, b.n_samples);
}
