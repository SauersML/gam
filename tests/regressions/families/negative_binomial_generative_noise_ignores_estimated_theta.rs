//! Regression for #1124: the generative observation model — `gam generate`,
//! `Model.sample_replicates`, `posterior_predictive_check` — must draw
//! Negative-Binomial responses with the **estimated** overdispersion `theta_hat`,
//! not the construction seed `theta = 1.0`. With the seed, synthetic / replicate
//! counts carry `Var(y) = mu + mu^2` instead of the correct
//! `Var(y) = mu + mu^2/theta_hat` — far too much overdispersion — and the
//! posterior-predictive p-values are wrong.
//!
//! This is the NB sibling of the already-fixed Beta #770 / Tweedie #771 /
//! Gamma #678 generative-dispersion bugs. The shared *root cause* was that the
//! dispersion-picking logic ("given a fit, what scalar dispersion feeds the
//! generative `NoiseModel`?") was **duplicated** across the CLI generate path
//! and the Python `sample_replicates` path — and a third, dead copy — so fixing
//! one copy left the others drawing at the seed. The pickers are now unified
//! into the single `gam::generative::family_noise_parameter`, which both live
//! paths call; this test pins that unified picker on a real NB fit.
//!
//! ## What is asserted (two independent angles)
//!
//!  1. *the fit records the overdispersion*: a real NB fit to data with true
//!     `theta = 3` records `theta_hat` well above the seed `1` in its scale
//!     metadata (`EstimatedNegBinTheta`).
//!  2. *the unified picker → noise composition threads it*: even when the family
//!     spec handed to the picker carries the un-refreshed seed `theta = 1`
//!     (the worst case the generate path can present), the canonical
//!     `family_noise_parameter` recovers `theta_hat` off the fit's scale
//!     metadata, and `NoiseModel::from_likelihood` then carries `theta_hat` for
//!     every row — never the seed.

use csv::StringRecord;
use gam::generative::{NoiseModel, family_noise_parameter};
use gam::types::LikelihoodSpec;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

/// One Negative-Binomial(mu, theta) count via the gamma-Poisson mixture:
/// `lambda ~ Gamma(shape=theta, scale=mu/theta)` then `y ~ Poisson(lambda)`,
/// giving `E[y]=mu`, `Var(y)=mu + mu^2/theta`.
fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng).max(1e-12);
    Poisson::new(lambda)
        .expect("poisson rate valid")
        .sample(rng)
}

#[test]
fn negative_binomial_generative_noise_uses_estimated_theta_not_seed() {
    init_parallelism();

    // Overdispersed NB data with true theta = 3 (so the seed theta = 1 is a
    // grossly different, easily-distinguished value). log mu = 0.5 + 0.3 x.
    const N: usize = 2000;
    const TRUE_THETA: f64 = 3.0;
    let mut rng = StdRng::seed_from_u64(5);
    let xdist = Uniform::new(0.0_f64, 4.0).expect("uniform support valid");
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = xdist.sample(&mut rng);
        let mu = (0.5 + 0.3 * xi).exp();
        x.push(xi);
        y.push(sample_negbin(mu, TRUE_THETA, &mut rng));
    }

    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| StringRecord::from(vec![(y[i] as i64).to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");

    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ smooth(x)", &ds, &cfg).expect("gam negbin fit should succeed")
    else {
        panic!("negative-binomial GLM should produce a Standard fit");
    };

    // Angle 1: the fit recorded the overdispersion theta_hat ~ 3, far above the
    // seed theta = 1. (If this fails the test is mis-set-up, not the bug.)
    let theta_hat = fit
        .fit
        .likelihood_scale
        .negbin_theta()
        .expect("NB fit must record an estimated theta in likelihood_scale");
    assert!(
        theta_hat > 1.8,
        "NB fit failed to recover the overdispersion (theta_hat={theta_hat}, true 3)"
    );

    // Angle 2: feed the canonical picker the WORST case the generate path could
    // present — a family spec still carrying the un-refreshed seed theta = 1 —
    // alongside the fit's scale metadata. The unified picker must recover
    // theta_hat off the scale metadata, not echo the seed.
    let seed_spec = LikelihoodSpec::negative_binomial_log(1.0);
    let picked = family_noise_parameter(
        fit.fit.likelihood_scale,
        fit.fit.standard_deviation,
        &seed_spec,
    )
    .expect("NB generative dispersion must resolve");
    assert!(
        (picked - theta_hat).abs() <= 1e-9 * theta_hat.max(1.0),
        "unified picker returned {picked} (seed theta=1?), expected theta_hat={theta_hat} (#1124)"
    );

    // And the composed NoiseModel — exactly what `gam generate` builds — must
    // carry theta_hat for every row, not the seed.
    let noise = NoiseModel::from_likelihood(&seed_spec, N, Some(picked))
        .expect("NB generative noise model builds");
    let NoiseModel::NegativeBinomial { theta } = noise else {
        panic!("expected a NegativeBinomial observation noise model");
    };
    assert_eq!(theta.len(), N, "per-row theta must span all rows");
    assert!(
        theta
            .iter()
            .all(|&t| (t - theta_hat).abs() <= 1e-9 * theta_hat.max(1.0)),
        "NB generative noise draws at the seed theta=1, not the fitted theta_hat={theta_hat} \
         (#1124): Var would be mu+mu^2 instead of mu+mu^2/theta_hat"
    );
}
