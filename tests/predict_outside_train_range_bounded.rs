//! Failing-ticket regression: predicting at x just outside the training
//! range must not produce wildly unbounded values. Smooth families differ
//! in their extrapolation behavior — but for a `s(x)` fit on x ∈ [0, 1]
//! with truth sin(2π x), predicting at x = 1.10 (10% past the boundary)
//! should yield a value within ±2 (= 1 × truth peak-to-peak above zero).
//!
//! A naive extrapolation that blows up (|fit(1.10)| > 5) indicates either
//! a polynomial-blowup in the basis tail or a numerical issue in the
//! frozen spec.

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
    let result = fit_from_formula(formula, data, &cfg).expect("fit");
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
fn small_extrapolation_predictions_are_bounded() {
    init_parallelism();
    let data = build_data(240, 0.05, 251);
    // 10% past each boundary
    let probes: Vec<f64> = vec![-0.10, -0.05, 1.05, 1.10];

    let cases: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
        ("s_default", "s(x)"),
    ];

    let bound = 5.0_f64; // truth peak ±1; allow 5x slack for extrapolation
    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_predict(&format!("y ~ {body}"), &data, &probes);
        for (i, &xt) in probes.iter().enumerate() {
            let v = yhat[i];
            eprintln!("[extrap-bounded] {label:10} x={xt:+.2} pred={v:+.3}");
            if !v.is_finite() {
                violations.push(format!("{label}: pred at x={xt} is non-finite ({v})"));
            } else if v.abs() > bound {
                violations.push(format!(
                    "{label}: |pred at x={xt}| = {:.3} > {bound} (truth peak ≈ 1)",
                    v.abs()
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "small-extrapolation predictions blew up beyond truth peak:\n  - {}",
        violations.join("\n  - "),
    );
}
