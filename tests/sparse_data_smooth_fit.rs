//! Failing-ticket regression: a 1D smooth fit on sparse but well-spaced
//! data (n = 30 with uniform coverage) must recover a moderate-frequency
//! truth without collapse.
//!
//! Setup: 30 well-spaced points on x ∈ [0, 1], y = sin(2π·2·x) + ε,
//! σ = 0.05. With 2 cycles and 30 points (~15 points per cycle) every
//! reasonable smoother should recover the truth with predicted span ≥ 1.6
//! (= 80% of truth peak-to-peak 2.0) and RMSE ≤ 0.15.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn build_data(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let x: Vec<f64> = (0..n)
        .map(|i| 0.02 + 0.96 * i as f64 / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin() + noise.sample(&mut rng))
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
    let result = fit_from_formula(formula, data, &cfg).expect("sparse fit");
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

fn span(v: &[f64]) -> f64 {
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
    mx - mn
}

#[test]
fn sparse_30_points_sin2_recovers_oscillations() {
    init_parallelism();
    let data = build_data(30, 127);
    let x_test: Vec<f64> = (0..400).map(|i| 0.02 + 0.96 * i as f64 / 399.0).collect();
    let y_truth: Vec<f64> = x_test
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin())
        .collect();

    let cases: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
        ("s_default", "s(x)"),
    ];

    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_predict(&format!("y ~ {body}"), &data, &x_test);
        let r = rmse(&yhat, &y_truth);
        let s = span(&yhat);
        eprintln!("[sparse-30] {label:10} rmse={r:.4} span={s:.3}");
        if r > 0.15 {
            violations.push(format!("{label}: rmse {r:.4} > 0.15"));
        }
        if s < 1.6 {
            violations.push(format!(
                "{label}: span {s:.3} < 1.6 (oscillations collapsed at n=30)"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "sparse n=30 fit fails to recover sin2 oscillations:\n  - {}",
        violations.join("\n  - "),
    );
}
