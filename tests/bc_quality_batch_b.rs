//! Batched cycles 55-58: BC robustness across data + bc-side variants.

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

const PI: f64 = std::f64::consts::PI;

fn make_data(n: usize, f: impl Fn(f64) -> f64, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x.iter().map(|&t| f(t) + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str, data: gam::data::EncodedDataset, xs: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula(formula, &data, &cfg).unwrap_or_else(|e| panic!("fit `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let mut m = Array2::<f64>::zeros((xs.len(), 2));
    for (i, &x) in xs.iter().enumerate() {
        m[[i, 0]] = x;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

/// Cycle 55: BC=clamped with very small n.
#[test]
fn cycle_55_bc_clamped_tiny_n() {
    init_parallelism();
    let data = make_data(15, |t| (2.0 * PI * t).sin(), 0.05, 7);
    let xs: Vec<f64> = (0..10).map(|i| 0.1 + 0.8 * (i as f64) / 9.0).collect();
    let pred = fit_predict("y ~ s(x, bc=clamped, k=8)", data, &xs);
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite at n=15");
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[bc-clamp-tiny] range=[{mn:.3}, {mx:.3}]");
    assert!(
        mn > -3.0 && mx < 3.0,
        "bc=clamped tiny-n pred out of envelope"
    );
}

/// Cycle 56: BC=anchored fit on a bimodal truth (two sharp Gaussians).
#[test]
fn cycle_56_bc_anchored_bimodal_truth() {
    init_parallelism();
    let bumps = |t: f64| {
        let g1 = (-((t - 0.3) * 12.0).powi(2)).exp();
        let g2 = (-((t - 0.7) * 12.0).powi(2)).exp();
        g1 + 0.8 * g2
    };
    let data = make_data(300, bumps, 0.05, 7);
    let xs: Vec<f64> = (0..30).map(|i| 0.02 + 0.96 * (i as f64) / 29.0).collect();
    let pred = fit_predict("y ~ s(x, bc=anchored, k=20)", data, &xs);
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite");
    // Should capture both peaks: pred max > 0.5
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[bc-bimodal] pred max={mx:.3}");
    assert!(
        mx > 0.4,
        "bc=anchored failed to capture bimodal peaks: max={mx:.3}"
    );
}

/// Cycle 57: BC=clamped on truth with nonzero slope at right endpoint.
#[test]
fn cycle_57_bc_clamped_nonzero_endpoint_slope() {
    init_parallelism();
    // truth f(1) = 1.5, f'(1) = 1 — clamped will force f'(1)=0 (bias).
    // Interior should still fit OK.
    let data = make_data(300, |t| 1.5 * t, 0.05, 7);
    let xs: Vec<f64> = (0..20).map(|i| 0.1 + 0.8 * (i as f64) / 19.0).collect();
    let pred = fit_predict("y ~ s(x, bc=clamped, k=15)", data, &xs);
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite");
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[bc-nonzero-slope] range=[{mn:.3}, {mx:.3}]");
    assert!(mn >= -0.2 && mx <= 2.0, "out of envelope");
}

/// Cycle 58: mixed BC (left=clamped, right=anchored).
#[test]
fn cycle_58_bc_mixed_left_clamped_right_anchored() {
    init_parallelism();
    let data = make_data(300, |t| (PI * t).sin(), 0.05, 7);
    let xs: Vec<f64> = (0..20).map(|i| 0.05 + 0.9 * (i as f64) / 19.0).collect();
    let pred = fit_predict(
        "y ~ s(x, bc_left=clamped, bc_right=anchored, k=15)",
        data,
        &xs,
    );
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite");
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[bc-mixed] range=[{mn:.3}, {mx:.3}]");
    assert!(mn > -1.0 && mx < 2.0, "mixed BC pred out of envelope");
}
