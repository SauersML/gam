//! Bug-hunt lock: a fitted Beta-regression's **response-scale observation
//! prediction interval** must use the *estimated* precision `φ̂`, not the
//! construction-time seed `φ`.
//!
//! Beta's precision is estimated jointly with the mean (issues #567/#769), and
//! its value is recorded in `likelihood_scale` (`EstimatedBetaPhi { phi }`),
//! NOT on the family enum `ResponseFamily::Beta { phi }` — that enum field stays
//! at the un-updated seed (default `1.0`). #770 fixed the *generate*/sampling
//! consumer to read the estimated `φ̂`; the **observation-interval** consumer in
//! `inference::predict` was missed.
//!
//! For a Beta response `Var(y) = μ(1−μ)/(1+φ)`, so the predictive band is
//! `μ̂ ± z·√(SE(μ̂)² + μ̂(1−μ̂)/(1+φ̂))`. The Tweedie and Gamma arms of the same
//! function read the fitted dispersion via `source.observation_phi()`
//! (= `likelihood_scale.fixed_phi()`); the Beta arm instead reads `*phi` off
//! `spec.response`, i.e. the seed. With the seed `φ=1`, the response-noise term
//! is `μ(1−μ)/2` — for high-precision data (large true `φ`) that is *enormously*
//! too wide: the interval ignores how concentrated the data actually is.
//!
//! This test fits high-precision Beta data (true `φ=30`, so `φ̂ ≫ 1`), takes the
//! observation interval, and back-solves the precision the interval's width
//! implies. A correct fit yields an implied precision ≈ `φ̂` (the wide band
//! collapses to the data's real noise); the seed-`φ` bug yields an implied
//! precision pinned near `1.0`.

use csv::StringRecord;
use gam::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Beta, Distribution, Uniform};

const Z95: f64 = 1.959_963_984_540_054; // qnorm(0.975)
const TRUE_PHI: f64 = 30.0;

fn logistic(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[test]
fn beta_observation_interval_uses_estimated_phi_not_seed() {
    init_parallelism();

    // High-precision heteroscedastic-free Beta data: μ(x) = logistic(0.3 + 1.1 x),
    // y ~ Beta(μφ, (1−μ)φ) with a large true φ so the data sit tightly around μ.
    let n = 3000usize;
    let mut rng = StdRng::seed_from_u64(0xB57A_u64);
    let ux = Uniform::new(-1.5_f64, 1.5_f64).expect("uniform");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = logistic(0.3 + 1.1 * xi);
            let a = mu * TRUE_PHI;
            let b = (1.0 - mu) * TRUE_PHI;
            Beta::new(a, b)
                .expect("beta params")
                .sample(&mut rng)
                .clamp(1.0e-6, 1.0 - 1.0e-6)
        })
        .collect();

    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode beta data");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ x", &ds, &cfg).expect("beta fit should succeed")
    else {
        panic!("expected a Standard Beta fit");
    };

    let phi_hat = fit
        .fit
        .likelihood_scale
        .fixed_phi()
        .expect("Beta fit must record an estimated precision");
    assert!(
        phi_hat > 5.0,
        "test mis-set-up: high-precision data should estimate φ̂ ≫ 1, got {phi_hat}"
    );

    // Predict the observation interval on an interior grid.
    let eval: Vec<f64> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let m = eval.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &xi) in eval.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design at eval grid");
    let dense = design.design.to_dense();

    let beta_logit = LikelihoodSpec::new(
        ResponseFamily::Beta { phi: 1.0 }, // the *seed* the saved family carries
        InverseLink::Standard(StandardLink::Logit),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        beta_logit,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: true,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("beta predict with observation interval");

    let lower = pred.observation_lower.expect("observation interval");
    let upper = pred.observation_upper.expect("observation interval");
    let mean = &pred.mean;
    let mean_se = &pred.mean_standard_error;

    // Back-solve the precision implied by each interval's width:
    //   hw = z·√(SE(μ̂)² + μ̂(1−μ̂)/(1+φ_implied))
    //   ⇒ 1+φ_implied = μ̂(1−μ̂) / ((hw/z)² − SE(μ̂)²).
    let mut implied_phis = Vec::with_capacity(m);
    for i in 0..m {
        let hw = 0.5 * (upper[i] - lower[i]);
        assert!(hw.is_finite() && hw > 0.0, "non-finite/zero obs half-width");
        let obs_var = (hw / Z95).powi(2);
        let response_var = (obs_var - mean_se[i].powi(2)).max(1e-12);
        let mu = mean[i];
        let one_plus_phi = (mu * (1.0 - mu) / response_var).max(1e-9);
        implied_phis.push(one_plus_phi - 1.0);
    }
    let implied_phi = implied_phis.iter().sum::<f64>() / implied_phis.len() as f64;

    eprintln!(
        "[beta-obs] estimated φ̂={phi_hat:.3} (true {TRUE_PHI}); observation-interval \
         implied precision ≈ {implied_phi:.3} (seed-φ bug pins this near 1.0)"
    );

    // The interval must reflect the *estimated* precision, not the seed φ=1.
    // A relative match to φ̂ is the precise contract; the seed bug lands the
    // implied precision near 1.0 (off by ~φ̂).
    let rel_err = (implied_phi - phi_hat).abs() / phi_hat;
    assert!(
        rel_err < 0.10,
        "Beta observation interval ignores the estimated precision: implied φ ≈ \
         {implied_phi:.3} but the fit estimated φ̂ = {phi_hat:.3} (rel err {rel_err:.3}). \
         The interval is using the seed φ = 1.0 (Var = μ(1−μ)/2), so it is √((1+φ̂)/2) \
         ≈ {:.1}× too wide on this high-precision data.",
        ((1.0 + phi_hat) / 2.0).sqrt()
    );
}
