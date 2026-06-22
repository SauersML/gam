//! Failing-ticket regression: no smooth family in `{matern, duchon, smooth}`
//! recovers a sharp step truth `1[x > 0.5]` to within a generous 0.50 max-error
//! envelope at moderate noise (σ=0.10, n=240, dense test grid).
//!
//! The adversarial 1D sweep flags every single family on the `step` truth:
//! the Gibbs-like overshoot near x=0.5 reaches ~0.6–0.7. While a step truth
//! is intrinsically hard for any global-bandwidth smoother, a sane smoother
//! should still keep the maximum absolute error below ~50% of the truth
//! peak-to-peak (= 0.50). Several literature smoothers achieve this; ours
//! does not. This test asserts the bound.

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

fn make_step_dataset(sigma: f64, n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y_noisy: Vec<f64> = x
        .iter()
        .map(|&t| if t > 0.5 { 1.0 } else { 0.0 } + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode step dataset")
}

fn fit_and_predict(formula: &str, data: &gam::data::EncodedDataset, x_test: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("step fit succeeded");
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
fn step_function_max_error_within_50pct_budget() {
    init_parallelism();
    let data = make_step_dataset(0.10, 240, 11);
    // Avoid evaluating exactly at the discontinuity x=0.5
    let x_test: Vec<f64> = (0..400)
        .map(|i| 0.001 + 0.998 * i as f64 / 399.0)
        .filter(|&t| (t - 0.5).abs() > 0.02)
        .collect();
    let y_truth_test: Vec<f64> = x_test
        .iter()
        .map(|&t| if t > 0.5 { 1.0 } else { 0.0 })
        .collect();

    // Truth peak-to-peak = 1.0. Even allowing a generous 0.50 envelope
    // (50% of peak), every family in {matern, duchon, smooth} exceeds it
    // because of Gibbs-like overshoot at the boundary.
    let budget = 0.50_f64;
    let cases: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
    ];

    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_and_predict(&format!("y ~ {body}"), &data, &x_test);
        let m = max_abs_err(&yhat, &y_truth_test);
        eprintln!("[step-recovery] {label:8} max_err={m:.4} (budget={budget:.2})");
        if m > budget {
            violations.push(format!("{label}: max_err {m:.4} > 0.50"));
        }
    }
    assert!(
        violations.is_empty(),
        "step recovery is poor across smooth families:\n  - {}",
        violations.join("\n  - "),
    );
}
