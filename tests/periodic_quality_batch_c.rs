//! Batched cycles 59-62: periodic 1D B-spline robustness across truths.

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

const TAU: f64 = std::f64::consts::TAU;

fn make_data(
    n: usize,
    f: impl Fn(f64) -> f64,
    sigma: f64,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, TAU).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t.iter().map(|&x| f(x) + noise.sample(&mut rng)).collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t.iter().zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str, data: gam::data::EncodedDataset, ts: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((ts.len(), 2));
    for (i, &t) in ts.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

/// Cycle 59: periodic 1D with k=80 (very fine basis) on smooth truth.
#[test]
fn cycle_59_periodic_1d_very_fine_k() {
    init_parallelism();
    let data = make_data(500, |t| t.cos(), 0.05, 7);
    let probes: Vec<f64> = (0..20).map(|i| TAU * (i as f64) / 19.0).collect();
    let pred = fit_predict(
        "y ~ s(t, periodic=true, period=6.283185307179586, k=80)",
        data,
        &probes,
    );
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite at k=80");
    let truth: Vec<f64> = probes.iter().map(|t| t.cos()).collect();
    let rmse = {
        let s: f64 = pred.iter().zip(truth.iter()).map(|(p, t)| (p - t).powi(2)).sum();
        (s / pred.len() as f64).sqrt()
    };
    eprintln!("[per-k80] rmse={rmse:.4}");
    assert!(rmse < 0.05, "k=80 fit poor: rmse={rmse:.4}");
}

/// Cycle 60: periodic 1D recovering different harmonic frequencies.
#[test]
fn cycle_60_periodic_1d_multi_frequency() {
    init_parallelism();
    for n_freq in [1usize, 2, 4] {
        let f = move |t: f64| (n_freq as f64 * t).sin();
        let data = make_data(400, f, 0.05, 7);
        let probes: Vec<f64> = (0..40).map(|i| TAU * (i as f64) / 39.0).collect();
        let pred = fit_predict(
            "y ~ s(t, periodic=true, period=6.283185307179586, k=30)",
            data,
            &probes,
        );
        let truth: Vec<f64> = probes.iter().map(|&t| f(t)).collect();
        let s: f64 = pred.iter().zip(truth.iter()).map(|(p, t)| (p - t).powi(2)).sum();
        let rmse = (s / pred.len() as f64).sqrt();
        eprintln!("[per-freq n={n_freq}] rmse={rmse:.4}");
        assert!(rmse < 0.15, "n_freq={n_freq} rmse={rmse:.4} too large");
    }
}

/// Cycle 61: periodic 1D with sharp pulse truth.
#[test]
fn cycle_61_periodic_1d_sharp_pulse_truth_stable() {
    init_parallelism();
    let f = |t: f64| {
        let dt = (t - std::f64::consts::PI).rem_euclid(TAU);
        let centered = if dt > std::f64::consts::PI { dt - TAU } else { dt };
        (-(centered / 0.15).powi(2)).exp()
    };
    let data = make_data(400, f, 0.05, 7);
    let probes: Vec<f64> = (0..40).map(|i| TAU * (i as f64) / 39.0).collect();
    let pred = fit_predict(
        "y ~ s(t, periodic=true, period=6.283185307179586, k=40)",
        data,
        &probes,
    );
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite");
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!("[per-pulse] range=[{mn:.3}, {mx:.3}]");
    // Should capture the pulse (max > 0.4) without crazy oscillation
    assert!(mx > 0.4, "sharp pulse: max={mx:.3} too small (pulse not captured)");
    assert!(mn > -0.5, "sharp pulse: min={mn:.3} too small (Gibbs/oscillation)");
}

/// Cycle 62: periodic 1D with constant truth → fit should be near-flat.
#[test]
fn cycle_62_periodic_1d_constant_truth() {
    init_parallelism();
    let data = make_data(400, |_| 3.5, 0.1, 7);
    let probes: Vec<f64> = (0..30).map(|i| TAU * (i as f64) / 29.0).collect();
    let pred = fit_predict(
        "y ~ s(t, periodic=true, period=6.283185307179586, k=20)",
        data,
        &probes,
    );
    let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
    let var: f64 = pred.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / pred.len() as f64;
    let std = var.sqrt();
    eprintln!("[per-const] mean={mean:.3} std={std:.4}");
    assert!((mean - 3.5).abs() < 0.1, "constant truth fit drifted: mean={mean:.3}");
    assert!(std < 0.05, "constant truth fit overfit noise: std={std:.4}");
}
