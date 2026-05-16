//! Failing-ticket regression: every standard 1D smooth family must recover
//! a low-degree polynomial truth to near noise-floor accuracy. A cubic
//! polynomial is in the null space of most penalty terms (second derivative
//! squared), so REML can drive the smoothing parameter to zero with no
//! penalty cost — the fit should be effectively unbiased.
//!
//! Truth: f(x) = 0.5 + 1.2 x − 1.5 x² + 0.8 x³ on x ∈ [0, 1], σ = 0.05,
//! n = 240. Truth peak-to-peak ≈ 1.0. Sane fit should achieve
//! RMSE ≤ 0.025 (= σ/2) for every family — this is what an OLS cubic fit
//! gets on the same data.

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
    let f = |t: f64| 0.5 + 1.2 * t - 1.5 * t * t + 0.8 * t * t * t;
    let y: Vec<f64> = x.iter().map(|&t| f(t) + noise.sample(&mut rng)).collect();
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
    let result = fit_from_formula(formula, data, &cfg).expect("polynomial fit");
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
fn cubic_polynomial_truth_recovered_at_noise_floor() {
    init_parallelism();
    let data = build_data(240, 0.05, 53);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let f = |t: f64| 0.5 + 1.2 * t - 1.5 * t * t + 0.8 * t * t * t;
    let y_truth: Vec<f64> = x_test.iter().map(|&t| f(t)).collect();

    let cases: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
        ("s_default", "s(x)"),
    ];

    // A cubic polynomial sits in (or very near) the null space of every
    // standard smoothing penalty. REML should drive λ → 0 and recover the
    // truth to noise-floor / sqrt(n). With n=240 and σ=0.05, the noise-
    // limited RMSE bound is ~0.05 / sqrt(240/8) ≈ 0.009; allow 0.030
    // for any reasonable mid-basis fit.
    let budget = 0.030_f64;
    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_predict(&format!("y ~ {body}"), &data, &x_test);
        let r = rmse(&yhat, &y_truth);
        eprintln!("[poly] {label:10} rmse={r:.4}");
        if r > budget {
            violations.push(format!("{label}: rmse {r:.4} > {budget:.3}"));
        }
    }
    assert!(
        violations.is_empty(),
        "cubic polynomial truth not recovered to noise floor:\n  - {}",
        violations.join("\n  - "),
    );
}
