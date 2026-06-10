//! Bug hunt: a user-supplied **fixed** Negative-Binomial overdispersion `theta`
//! (`FitConfig::negative_binomial_theta`, exposed on the CLI as
//! `--negative-binomial-theta`, documented "Fixed size/overdispersion parameter
//! for `--family negative-binomial`") is silently ignored — `theta` is always
//! re-estimated from the data, so the flag is a complete no-op on the fit,
//! predictions, and standard errors.
//!
//! This is the COMPLEMENT of the now-fixed #802. #802 made `theta` *estimated by
//! default* (no longer frozen at the seed 1.0). But the fix wired the user knob
//! only as a discarded SEED: `LikelihoodSpec::default_scale_metadata`
//! (src/types.rs:1258-1260) unconditionally returns
//! `LikelihoodScaleMetadata::EstimatedNegBinTheta` for `NegativeBinomial`,
//! whatever value the user passed. The PIRLS driver then re-estimates whenever
//! `scale.negbin_theta_is_estimated()` is true (src/solver/pirls/mod.rs:2157),
//! overwriting the supplied value. There is no "fixed NB theta" scale variant, so
//! the documented *fixed* mode does not exist.
//!
//! Observable consequence (reproduced through the CLI): fitting the same dataset
//! with `--negative-binomial-theta 0.5`, `2.0`, and `50.0` produces bit-identical
//! coefficients, predicted means, and standard errors — every run re-estimates to
//! the same theta_hat ~ 1.99, while the saved model's `family_state` stores the
//! (ignored) user value, an internal inconsistency.
//!
//! ## What is asserted
//!
//! With NB data whose ML overdispersion is ~2.0, two fits that differ ONLY in the
//! supplied *fixed* theta (0.3 vs 30.0) must:
//!   1. honour each fixed value — the recorded theta equals the supplied one
//!      (a fixed parameter is held, not re-estimated); and
//!   2. therefore differ from each other.
//! Today both fits ignore the knob and record the same data-driven theta_hat, so
//! both assertions fail. When fixed-theta is honoured they pass unchanged.

use csv::StringRecord;
use gam::types::ResponseFamily;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

/// One Negative-Binomial(mu, theta) count via the gamma-Poisson mixture.
fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> i64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng).max(1e-12);
    Poisson::new(lambda)
        .expect("poisson rate valid")
        .sample(rng) as i64
}

/// Fit `y ~ x` as NB(log) with an explicitly FIXED theta and return the theta the
/// fit actually used (the recorded scale-metadata theta, which must equal the
/// family-variant theta).
fn fit_with_fixed_theta(x: &[f64], y: &[i64], fixed_theta: f64) -> f64 {
    let n = x.len();
    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");

    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        // The user pins the overdispersion: per the CLI help this is a
        // "Fixed size/overdispersion parameter".
        negative_binomial_theta: Some(fixed_theta),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ x", &ds, &cfg).expect("gam negbin fit should succeed")
    else {
        panic!("negative-binomial GLM should produce a Standard fit");
    };

    let scale_theta = fit
        .fit
        .likelihood_scale
        .negbin_theta()
        .expect("NB fit must record a theta in likelihood_scale");
    let family_theta = match fit
        .fit
        .likelihood_family
        .as_ref()
        .expect("NB fit must record a likelihood family")
        .response
    {
        ResponseFamily::NegativeBinomial { theta, .. } => theta,
        ref other => panic!("expected NegativeBinomial family, got {other:?}"),
    };
    assert!(
        (scale_theta - family_theta).abs() <= 1e-9 * scale_theta.max(1.0),
        "scale-metadata theta {scale_theta} and family-variant theta {family_theta} must agree"
    );
    scale_theta
}

#[test]
fn negative_binomial_fixed_theta_is_honoured_not_re_estimated() {
    init_parallelism();

    // Data whose ML overdispersion is ~2.0, far from both fixed values below, so
    // an "estimate anyway" path is unmistakably distinguishable from "hold fixed".
    let n = 1500usize;
    let mut rng = StdRng::seed_from_u64(0xBEEF_2026);
    let ux = Uniform::new(-2.0_f64, 2.0_f64).expect("uniform -2..2");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let true_theta = 2.0_f64;
    let y: Vec<i64> = x
        .iter()
        .map(|&xi| sample_negbin((0.6 + 0.5 * xi).exp(), true_theta, &mut rng))
        .collect();

    let fixed_small = 0.3_f64;
    let fixed_large = 30.0_f64;
    let theta_small = fit_with_fixed_theta(&x, &y, fixed_small);
    let theta_large = fit_with_fixed_theta(&x, &y, fixed_large);

    eprintln!(
        "[nb-fixed-theta] supplied fixed theta: small={fixed_small} large={fixed_large}; \
         recorded theta: small={theta_small:.5} large={theta_large:.5} \
         (data ML theta ~ {true_theta}). A no-op knob records the same estimate for both."
    );

    // (1) Each fixed value must be honoured (held, not re-estimated).
    assert!(
        (theta_small - fixed_small).abs() <= 1e-3 * fixed_small.max(1.0),
        "fixed negative_binomial_theta={fixed_small} was not honoured: the fit used \
         theta={theta_small:.5} (re-estimated from the data instead of held fixed)"
    );
    assert!(
        (theta_large - fixed_large).abs() <= 1e-3 * fixed_large.max(1.0),
        "fixed negative_binomial_theta={fixed_large} was not honoured: the fit used \
         theta={theta_large:.5} (re-estimated from the data instead of held fixed)"
    );

    // (2) Two different fixed thetas on the same data must therefore differ.
    assert!(
        (theta_small - theta_large).abs() > 1.0,
        "two wildly different fixed thetas ({fixed_small} vs {fixed_large}) produced the \
         same recorded theta ({theta_small:.5} vs {theta_large:.5}) — the \
         --negative-binomial-theta knob has no effect on the fit"
    );
}
