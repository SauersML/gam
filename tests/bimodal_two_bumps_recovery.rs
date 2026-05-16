//! Failing-ticket regression: a two-bump truth
//! `f(x) = exp(-(x-0.3)²/0.01) - exp(-(x-0.75)²/0.01)`  — one positive
//! bump and one negative bump well-separated — must be recovered with
//! correct sign at both peaks.
//!
//! Setup: σ = 0.05, n = 240. At x = 0.30 the prediction must be ≥ +0.50,
//! and at x = 0.75 it must be ≤ -0.50. Failure of either is a clear sign
//! the smoother over-regularized one or both peaks.

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

fn truth(t: f64) -> f64 {
    (-((t - 0.3).powi(2)) / 0.01).exp() - (-((t - 0.75).powi(2)) / 0.01).exp()
}

fn build_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
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
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str, data: &gam::data::EncodedDataset, x_test: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("two-bumps fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = x_test.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &t) in x_test.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn two_bumps_positive_and_negative_peaks_recovered() {
    init_parallelism();
    let data = build_data(240, 0.05, 211);
    let probes: Vec<f64> = vec![0.30, 0.75];

    let cases: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
        ("s_default", "s(x)"),
    ];

    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_predict(&format!("y ~ {body}"), &data, &probes);
        let pos = yhat[0];
        let neg = yhat[1];
        eprintln!(
            "[two-bumps] {label:10} pred(0.30)={pos:+.3} pred(0.75)={neg:+.3}"
        );
        if pos < 0.50 {
            violations.push(format!(
                "{label}: prediction at +bump (x=0.30) is {pos:+.3} < +0.50 (oversmoothed)"
            ));
        }
        if neg > -0.50 {
            violations.push(format!(
                "{label}: prediction at -bump (x=0.75) is {neg:+.3} > -0.50 (oversmoothed)"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "two-bumps truth not recovered with correct sign and amplitude:\n  - {}",
        violations.join("\n  - "),
    );
}
