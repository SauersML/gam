//! Failing-ticket regression: `duchon(x)` (any centers) badly oversmooths
//! sin(2π·8·x) at moderate noise (σ=0.10, n=240).
//!
//! Adversarial 1D quality sweep (`scripts/_sweep_1d_quality.py`) shows
//! Duchon with default settings, centers=20, and centers=50 all exceed a
//! generous max-error budget of 0.30 × peak-to-peak (= 0.60) on this truth,
//! while `matern(x)` and `smooth(x)` fit it comfortably. The Duchon
//! length-scale / spectral-init path likely picks too-wide a bandwidth and
//! oversmooths, so the recovered fit has the right span (~2) but is phase-
//! and amplitude-distorted across the high-frequency oscillations.
//!
//! A sane smooth should achieve max|ŷ − y_truth| ≤ 0.60 on sin8 at σ=0.10.
//! This test asserts that bound on all three Duchon variants.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_sin_dataset(
    freq: f64,
    sigma: f64,
    n: usize,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let two_pi_f = 2.0 * std::f64::consts::PI * freq;
    let y_noisy: Vec<f64> = x
        .iter()
        .map(|&t| (two_pi_f * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode sin dataset")
}

fn fit_and_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    x_test: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("duchon fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = x_test.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &t) in x_test.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let test_design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild design from frozen spec");
    test_design.design.apply(&fit.fit.beta).to_vec()
}

fn max_abs_err(yhat: &[f64], y: &[f64]) -> f64 {
    yhat.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn duchon_sin8_max_error_within_budget() {
    init_parallelism();
    let data = make_sin_dataset(8.0, 0.10, 240, 11);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth_test: Vec<f64> = x_test
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * 8.0 * t).sin())
        .collect();

    let cases: &[(&str, &str)] = &[
        ("duchon-default", "duchon(x)"),
        ("duchon-centers20", "duchon(x, centers=20)"),
        ("duchon-centers50", "duchon(x, centers=50)"),
    ];

    // Truth peak-to-peak is 2.0; 30% of that is 0.60. A capable 1D smooth
    // recovers sin8 to well under this at σ=0.10 (`matern` and `smooth`
    // hit ~0.25 max error on the same data).
    let budget = 0.60_f64;
    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_and_predict(&format!("y ~ {body}"), &data, &x_test);
        let m = max_abs_err(&yhat, &y_truth_test);
        eprintln!("[duchon-sin8] {label:18} max_err={m:.4}");
        if m > budget {
            violations.push(format!(
                "{label}: max_err {m:.4} > {budget:.2} (truth peak=2.0, 30% budget)"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "duchon family oversmooths sin8 at σ=0.10:\n  - {}",
        violations.join("\n  - "),
    );
}
