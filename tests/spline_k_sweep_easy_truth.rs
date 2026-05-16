//! Failing-ticket regression: `s(x, k=K)` quality must be robust to the
//! choice of K on an easy truth. REML controls effective degrees of
//! freedom; the basis budget K just provides headroom.
//!
//! Truth: sin(2π·x) on [0, 1], σ = 0.05, n = 240. For K ∈ {4, 6, 10, 20}
//! every fit should achieve RMSE ≤ 0.04 (≈ noise floor).

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
    let result = fit_from_formula(formula, data, &cfg).expect("spline-k fit");
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
fn spline_k_sweep_uniform_quality() {
    init_parallelism();
    let data = build_data(240, 0.05, 191);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth: Vec<f64> = x_test
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
        .collect();

    let ks = [4usize, 6, 10, 20];
    let budget = 0.04_f64;
    let mut violations = Vec::<String>::new();
    for &k in &ks {
        let formula = format!("y ~ s(x, k={k})");
        let yhat = fit_predict(&formula, &data, &x_test);
        let r = rmse(&yhat, &y_truth);
        eprintln!("[s-k] k={k:2} rmse={r:.4}");
        if r > budget {
            violations.push(format!("k={k}: rmse {r:.4} > {budget:.2}"));
        }
    }
    assert!(
        violations.is_empty(),
        "s(x, k=K) quality is not uniform across K:\n  - {}",
        violations.join("\n  - "),
    );
}
