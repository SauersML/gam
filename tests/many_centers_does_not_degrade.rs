//! Failing-ticket regression: increasing `centers` for `duchon(x, centers=K)`
//! must not dramatically degrade fit quality. More basis budget combined
//! with REML smoothing should be at worst weakly worse, never wildly worse.
//!
//! On the sweep, `duchon(x, centers=20)` produces max-err ≈ 1.35 on sin8
//! at σ=0.10 while `duchon(x, centers=50)` produces 0.90 and `duchon(x)`
//! (default centers) produces 0.93. The centers=20 result is the worst —
//! suggesting a basin-of-attraction issue in the center placement /
//! length-scale init at small-but-not-tiny center counts.
//!
//! We assert that for the smooth truth sin(2π·2·x) (which is easy), all
//! three center counts produce RMSE ≤ 0.10. Currently centers=20 produces
//! noticeably worse quality due to the same init pathology.

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

fn build_data(n: usize, sigma: f64, freq: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let two_pi_f = 2.0 * std::f64::consts::PI * freq;
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (two_pi_f * t).sin() + noise.sample(&mut rng))
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
    let result = fit_from_formula(formula, data, &cfg).expect("duchon fit");
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

fn rmse(yhat: &[f64], y: &[f64]) -> f64 {
    let n = y.len() as f64;
    (yhat
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

#[test]
fn duchon_centers_sweep_uniform_quality_on_easy_truth() {
    init_parallelism();
    // Easy truth: sin(2π·2·x) — 2 cycles on [0,1], very smooth.
    let data = build_data(240, 0.10, 2.0, 41);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth: Vec<f64> = x_test
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin())
        .collect();

    let cases: &[(&str, &str)] = &[
        ("centers20", "duchon(x, centers=20)"),
        ("centers50", "duchon(x, centers=50)"),
        ("centers100", "duchon(x, centers=100)"),
    ];

    // Budget: 2 cycles is trivial for any reasonable smooth, RMSE should
    // be close to σ=0.10. Each variant should hit ≤ 0.05.
    let budget = 0.05_f64;
    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_predict(&format!("y ~ {body}"), &data, &x_test);
        let r = rmse(&yhat, &y_truth);
        eprintln!("[duchon-centers] {label:10} rmse={r:.4}");
        if r > budget {
            violations.push(format!("{label}: rmse {r:.4} > {budget:.2}"));
        }
    }
    assert!(
        violations.is_empty(),
        "duchon centers sweep produced non-uniform quality on an easy truth:\n  - {}",
        violations.join("\n  - "),
    );
}
