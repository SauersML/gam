//! Regression: Matern with the default smoothness must not collapse on a
//! moderately-high-frequency truth.
//!
//! Before the data-aware length_scale auto-init, `matern(x)` defaulted its
//! length scale to 1.0 — a basin from which the REML optimizer could not
//! escape for ν ≥ 5/2. The fit silently collapsed to a near-constant
//! prediction (RMSE ≈ 0.71, span ≈ 0.08) on sin(2π·8·x). After the fix,
//! the planner overrides the length_scale=0.0 sentinel with
//! `data_range / sqrt(n)`, putting REML on the wiggly side of the basin so
//! the optimizer can find the correct length scale.

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
) -> (Vec<f64>, Vec<f64>, gam::data::EncodedDataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let two_pi_f = 2.0 * std::f64::consts::PI * freq;
    let y_truth: Vec<f64> = x.iter().map(|t| (two_pi_f * t).sin()).collect();
    let y_noisy: Vec<f64> = y_truth
        .iter()
        .map(|&v| v + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode sin dataset");
    (x, y_truth, data)
}

fn fit_and_predict(formula: &str, data: &gam::data::EncodedDataset, x_test: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("matern fit succeeded");
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
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

#[test]
fn matern_default_does_not_collapse_on_sin8() {
    init_parallelism();
    let (_x, _y_truth, data) = make_sin_dataset(8.0, 0.10, 240, 11);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth_test: Vec<f64> = x_test
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * 8.0 * t).sin())
        .collect();

    // Several ν values that previously collapsed with length_scale=1.0
    let cases: &[(&str, &str)] = &[
        ("nu=3/2", "matern(x, nu=3/2)"),
        ("nu=5/2", "matern(x, nu=5/2)"),
        ("nu=7/2", "matern(x, nu=7/2)"),
        ("nu=9/2", "matern(x, nu=9/2)"),
    ];

    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_and_predict(&format!("y ~ {body}"), &data, &x_test);
        let r = rmse(&yhat, &y_truth_test);
        let s = span(&yhat);
        eprintln!("[matern-init] {label:8} rmse={r:.4} span={s:.3}");
        // Truth span is ~2; sane fit should keep at least 75% of it.
        if s < 1.5 {
            violations.push(format!("{label}: span {s:.3} < 1.5 (collapsed)"));
        }
        // Truth-vs-fit RMSE budget: well above noise floor σ=0.10 but well
        // below the constant-fit RMSE of ~0.71 the bug produced.
        if r > 0.20 {
            violations.push(format!("{label}: rmse {r:.4} > 0.20"));
        }
    }
    assert!(
        violations.is_empty(),
        "matern default length_scale init regressed:\n  - {}",
        violations.join("\n  - "),
    );
}
