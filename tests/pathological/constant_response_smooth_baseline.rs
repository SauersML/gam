//! Failing-ticket regression: when the response y is constant (no signal),
//! a 1D smooth fit should produce predictions tightly clustered around
//! that constant. Predicted span on a dense grid must be ≤ a few σ.
//!
//! Setup: n=200 uniform x ∈ [0, 1], y = 0.7 + ε with σ = 0.05. The truth
//! is f(x) = 0.7. Any sane smooth should drive λ → ∞ and recover a
//! near-constant prediction with span ≤ 0.05. A failure here indicates
//! the smoothing-parameter optimizer is not properly going to the
//! infinite-smoothness limit.

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

fn build_data(n: usize, sigma: f64, c: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = (0..n).map(|_| c + noise.sample(&mut rng)).collect();
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
    let result = fit_from_formula(formula, data, &cfg).expect("constant-truth fit");
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

fn span(v: &[f64]) -> f64 {
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
    mx - mn
}

#[test]
fn constant_truth_predicted_span_is_tight() {
    init_parallelism();
    let data = build_data(200, 0.05, 0.7, 97);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let cases: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
        ("s_default", "s(x)"),
    ];
    // Truth has span 0. Predicted span should be at most a few times σ; we
    // allow 0.05 (= 1 × σ). Larger spans indicate REML failed to drive λ
    // toward infinity for a no-signal truth.
    let budget = 0.05_f64;
    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_predict(&format!("y ~ {body}"), &data, &x_test);
        let s = span(&yhat);
        let mean: f64 = yhat.iter().sum::<f64>() / yhat.len() as f64;
        eprintln!("[const-truth] {label:10} span={s:.4} mean={mean:.3}");
        if s > budget {
            violations.push(format!(
                "{label}: span {s:.4} > {budget:.2} (REML did not max-smooth a constant truth)"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "constant-truth fit has too-large prediction span:\n  - {}",
        violations.join("\n  - "),
    );
}
