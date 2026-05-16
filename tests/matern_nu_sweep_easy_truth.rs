//! Failing-ticket regression: `matern(x, nu=ν)` should produce
//! similar-quality fits across ν ∈ {1/2, 3/2, 5/2, 7/2, 9/2} for a smooth
//! truth that all ν values can represent.
//!
//! Truth: f(x) = sin(2π·x), σ = 0.05, n = 240. RMSE budget is 0.10 (~2σ).
//! Several ν values previously collapsed (RMSE > 0.5) due to a basin
//! / length-scale init issue. The data-aware length_scale auto-init fixed
//! some ν values, but not all. This test asserts uniform quality across
//! all five ν values.

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

fn build_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
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
    let result = fit_from_formula(formula, data, &cfg).expect("matern nu fit");
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
fn matern_nu_sweep_uniform_quality_on_sin1() {
    init_parallelism();
    let data = build_data(240, 0.05, 137);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth: Vec<f64> = x_test
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
        .collect();

    let cases: &[(&str, &str)] = &[
        ("nu=1/2", "matern(x, nu=1/2)"),
        ("nu=3/2", "matern(x, nu=3/2)"),
        ("nu=5/2", "matern(x, nu=5/2)"),
        ("nu=7/2", "matern(x, nu=7/2)"),
        ("nu=9/2", "matern(x, nu=9/2)"),
    ];

    // sin(2π x) is in C^∞ and exactly representable by any Matern of any ν
    // (in the basis-completion limit). RMSE should be at noise-floor
    // ~σ/√(n/k) ~ 0.05/√24 ≈ 0.01 for any well-tuned fit. Budget 0.08 to
    // accommodate any tightness loss from k or length_scale init.
    let budget = 0.08_f64;
    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_predict(&format!("y ~ {body}"), &data, &x_test);
        let r = rmse(&yhat, &y_truth);
        eprintln!("[matern-nu] {label:8} rmse={r:.4}");
        if r > budget {
            violations.push(format!("{label}: rmse {r:.4} > {budget:.2}"));
        }
    }
    assert!(
        violations.is_empty(),
        "matern ν sweep not uniformly good on easy sin1 truth:\n  - {}",
        violations.join("\n  - "),
    );
}
