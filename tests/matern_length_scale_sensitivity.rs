//! Failing-ticket regression: `matern(x, length_scale=L)` fits should be
//! roughly stable across a wide band of explicit length scales for a smooth
//! truth, because REML can pick its own smoothing parameter on top.
//!
//! Truth = sin(2π x), σ=0.10, n=240. We try `length_scale` values in
//! {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}. A capable Matern smoother should
//! recover the truth with RMSE ≤ 0.10 for every one of these — the
//! length_scale just sets the bandwidth basin, and REML should refine.
//!
//! Currently several extreme length_scale values produce dramatically bad
//! fits (rmse > 0.3) on a truth that a sane fit nails to noise-floor.

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

fn build_data(sigma: f64, n: usize, seed: u64) -> gam::data::EncodedDataset {
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
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild design");
    design.design.apply(&fit.fit.beta).to_vec()
}

fn rmse(yhat: &[f64], y: &[f64]) -> f64 {
    let n = y.len() as f64;
    (yhat
        .iter()
        .zip(y)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

#[test]
fn matern_length_scale_sweep_stays_within_budget() {
    init_parallelism();
    let data = build_data(0.10, 240, 19);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth: Vec<f64> = x_test
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
        .collect();

    // Length scales spanning two orders of magnitude. A robust matern fit
    // should drive REML to find the same effective smoothing regardless of
    // this initial bandwidth — RMSE should be in the same noise-limited band.
    let length_scales = [0.02, 0.05, 0.10, 0.25, 0.50, 1.0];
    // Budget: 0.10 ~ σ. This is what good 1D smooth-family fits achieve on
    // sin(2πx) at σ=0.10 (matern default = 0.025, smooth = 0.020).
    let budget = 0.10_f64;

    let mut violations = Vec::<String>::new();
    for &ls in &length_scales {
        let formula = format!("y ~ matern(x, length_scale={ls})");
        let yhat = fit_predict(&formula, &data, &x_test);
        let r = rmse(&yhat, &y_truth);
        eprintln!("[matern-ls] length_scale={ls:.2} rmse={r:.4}");
        if r > budget {
            violations.push(format!("length_scale={ls}: rmse {r:.4} > {budget:.2}"));
        }
    }
    assert!(
        violations.is_empty(),
        "matern length_scale sweep produced non-uniform quality (REML is not \
         refining beyond the initial bandwidth):\n  - {}",
        violations.join("\n  - "),
    );
}
