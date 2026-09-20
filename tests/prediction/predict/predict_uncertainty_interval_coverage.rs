//! Calibration gate: the 95% conditional-covariance interval for the mean of a
//! smooth 1-D Gaussian fit must cover the noise-free truth at its NOMINAL rate,
//! neither below it (under-stated uncertainty) nor above it (inflated SEs,
//! i.e. conservative intervals, which SPEC treats as a bug too).
//!
//! ## What is measured
//!
//! Each replicate fits `y ~ smooth(x)` to `sin(2πx) + N(0, σ²)` and records its
//! across-the-function coverage (ACP): the fraction of 200 held-out interior
//! grid points whose `mean ± 1.96·SE` interval contains the truth. For the
//! Bayesian (conditional) covariance of a penalized smoother, ACP is nominal in
//! expectation (Nychka 1988; Marra & Wood 2012), so the replicate MEAN of ACP
//! must sit at 0.95.
//!
//! ## Why the bar is what it is
//!
//! A single fit's ACP is not a binomial proportion over 200 independent points:
//! the 200 indicators share one fitted curve, so one replicate can legitimately
//! land anywhere from ~0.7 to 1.0. The old single-seed `coverage >= 0.80` bar
//! was therefore both underived and one-sided: an interval with every SE
//! doubled covers at ~1.0 and passed.
//!
//! The replicates are independent, so the Monte-Carlo standard error of the
//! mean ACP is `sd(ACP)/√R`, taken from the replicates themselves. The gate is
//! the two-sided `|mean ACP − 0.95| ≤ 3·sd/√R`. Nothing in it is tuned: the
//! centre is the nominal level and the width is the sampling error of this
//! fixture's own estimate.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const NOMINAL: f64 = 0.95;
/// Two-sided 95% normal quantile, matching `NOMINAL`.
const Z_NOMINAL: f64 = 1.96;
const SIGMA: f64 = 0.10;
const N_TRAIN: usize = 240;
const N_TEST: usize = 200;
const REPLICATES: u64 = 40;

fn truth(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}

/// One replicate's across-the-function coverage of the 95% mean interval.
fn replicate_acp(seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");

    let mut x: Vec<f64> = (0..N_TRAIN).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ smooth(x)", &data, &cfg)
        .unwrap_or_else(|e| panic!("seed {seed}: Gaussian smooth fit must succeed: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("seed {seed}: expected a standard fit")
    };

    // Held-out points across the interior of the training range.
    let mut new_data = Array2::<f64>::zeros((N_TEST, 2));
    let mut truth_at = Array1::<f64>::zeros(N_TEST);
    for i in 0..N_TEST {
        let xt = 0.02 + 0.96 * (i as f64) / ((N_TEST - 1) as f64);
        new_data[[i, 0]] = xt;
        new_data[[i, 1]] = 0.0;
        truth_at[i] = truth(xt);
    }
    let test_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("rebuild predict design");
    let mean = test_design.design.apply(&fit.fit.beta);

    let cov = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("standard Gaussian fit should expose a conditional covariance");

    let mut covered = 0usize;
    for i in 0..N_TEST {
        // Row i of the test design is Xᵀ e_i.
        let mut e = Array1::<f64>::zeros(N_TEST);
        e[i] = 1.0;
        let xi = test_design.design.apply_transpose(&e);
        let var = xi.dot(&cov.dot(&xi));
        assert!(
            var.is_finite() && var > 0.0,
            "seed {seed}: prediction variance at test point {i} must be finite and positive, \
             got {var}"
        );
        let se = var.sqrt();
        if (truth_at[i] - mean[i]).abs() <= Z_NOMINAL * se {
            covered += 1;
        }
    }
    covered as f64 / N_TEST as f64
}

#[test]
fn smooth_1d_95pct_predictive_interval_covers_truth() {
    init_parallelism();
    let acp: Vec<f64> = (0..REPLICATES).map(|r| replicate_acp(31 + r)).collect();

    let r = acp.len() as f64;
    let mean = acp.iter().sum::<f64>() / r;
    let var = acp.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (r - 1.0);
    let mc_se = (var / r).sqrt();
    let band = 3.0 * mc_se;
    eprintln!(
        "[predict-coverage] replicates={REPLICATES} n_train={N_TRAIN} n_test={N_TEST} \
         σ={SIGMA} mean_acp={mean:.4} sd_acp={:.4} mc_se={mc_se:.4} band=±{band:.4} \
         per_replicate={acp:.3?}",
        var.sqrt()
    );

    assert!(
        (mean - NOMINAL).abs() <= band,
        "95% mean interval is miscalibrated: mean across-the-function coverage {mean:.4} over \
         {REPLICATES} replicates, nominal {NOMINAL}, allowed ±{band:.4} (3 Monte-Carlo SEs). \
         Below the band the SEs are understated; above it they are inflated."
    );
}
