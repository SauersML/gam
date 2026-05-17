//! Batched cycles 67-70: matern smooth quality across nu and length scales.

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

fn make_dataset(
    n: usize,
    f: impl Fn(f64) -> f64,
    sigma: f64,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x.iter().map(|&t| f(t) + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x.iter().zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str, data: gam::data::EncodedDataset, xs: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((xs.len(), 2));
    for (i, &x) in xs.iter().enumerate() {
        m[[i, 0]] = x;
        m[[i, 1]] = 0.0;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

/// Cycle 67: matern with nu=1/2 on a step-function truth.
#[test]
fn cycle_67_matern_half_step_truth_stable() {
    init_parallelism();
    let data = make_dataset(300, |t| if t < 0.5 { 0.0 } else { 1.0 }, 0.05, 7);
    let xs: Vec<f64> = (0..40).map(|i| 0.02 + 0.96 * (i as f64) / 39.0).collect();
    let pred = fit_predict("y ~ matern(x, nu=1/2, k=20)", data, &xs);
    assert!(pred.iter().all(|v| v.is_finite()));
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[matern-half-step] range=[{mn:.3}, {mx:.3}]");
    assert!(mn > -0.5 && mx < 1.5, "step fit out of envelope");
}

/// Cycle 68: matern centered on sinusoid; recovery RMSE.
#[test]
fn cycle_68_matern_sinusoid_recovery_rmse() {
    init_parallelism();
    for nu in ["3/2", "5/2", "7/2"] {
        let data = make_dataset(300, |t| (2.0 * std::f64::consts::PI * t).sin(), 0.05, 7);
        let xs: Vec<f64> = (0..40).map(|i| 0.02 + 0.96 * (i as f64) / 39.0).collect();
        let pred = fit_predict(&format!("y ~ matern(x, nu={nu})"), data, &xs);
        let truth: Vec<f64> = xs
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let s: f64 = pred.iter().zip(truth.iter()).map(|(p, t)| (p - t).powi(2)).sum();
        let rmse = (s / pred.len() as f64).sqrt();
        eprintln!("[matern-sin nu={nu}] rmse={rmse:.4}");
        assert!(rmse < 0.1, "matern nu={nu} rmse={rmse:.4} too large");
    }
}

/// Cycle 69: matern with explicit length_scale matches auto-fit reasonably.
#[test]
fn cycle_69_matern_explicit_vs_auto_length_scale() {
    init_parallelism();
    let xs: Vec<f64> = (0..30).map(|i| 0.05 + 0.9 * (i as f64) / 29.0).collect();
    let mk_data = || make_dataset(300, |t: f64| (2.0 * std::f64::consts::PI * t).sin(), 0.05, 7);
    let pred_auto = fit_predict("y ~ matern(x, nu=5/2)", mk_data(), &xs);
    let pred_ls = fit_predict("y ~ matern(x, nu=5/2, length_scale=0.2)", mk_data(), &xs);
    let truth: Vec<f64> = xs
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
        .collect();
    let rmse = |p: &[f64]| -> f64 {
        let s: f64 = p.iter().zip(truth.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        (s / p.len() as f64).sqrt()
    };
    eprintln!(
        "[matern-ls] auto rmse={:.4} ls=0.2 rmse={:.4}",
        rmse(&pred_auto),
        rmse(&pred_ls)
    );
    assert!(rmse(&pred_auto) < 0.05, "auto rmse too large");
    assert!(rmse(&pred_ls) < 0.10, "explicit ls rmse too large");
}

/// Cycle 70: matern at the lowest supported k (basis on the edge).
#[test]
fn cycle_70_matern_minimum_centers() {
    init_parallelism();
    for k in [5usize, 8, 12] {
        let data = make_dataset(200, |t: f64| t.powi(3), 0.05, 7);
        let xs: Vec<f64> = (0..20).map(|i| 0.05 + 0.9 * (i as f64) / 19.0).collect();
        let pred = fit_predict(
            &format!("y ~ matern(x, nu=3/2, centers={k})"),
            data,
            &xs,
        );
        assert!(pred.iter().all(|v| v.is_finite()), "k={k} non-finite");
        let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        eprintln!("[matern-k{k}] range=[{mn:.3}, {mx:.3}]");
        assert!(mn > -1.0 && mx < 2.0, "k={k} pred out of envelope");
    }
}
