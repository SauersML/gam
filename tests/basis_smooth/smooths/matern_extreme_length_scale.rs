//! Matérn with extreme fixed length scales: very small (a near-delta kernel)
//! and very large (a near-polynomial kernel). A positive finite scale is a
//! well-posed request, so a refusal is a defect, and the fit is scored
//! against the known truth by its own posterior band around the basis's
//! best approximation (#4377). A non-positive scale is an invalid request
//! and must be refused as a formula error.

#[path = "../../common/misc/smooth_truth_scoring.rs"]
mod smooth_truth_scoring;

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::{
    ErrorCategory, FitConfig, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use smooth_truth_scoring::{fit_and_score, probe_matrix};

fn truth(x: f64) -> f64 {
    (4.0 * std::f64::consts::PI * x).sin()
}

/// Fixture `y = sin(4πx) + N(0, 0.05²)`; returns the data and the
/// noise-free truth at the training rows.
fn make_dataset(n: usize) -> (EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let f: Vec<f64> = x.iter().map(|&t| truth(t)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(f.iter())
        .map(|(a, fa)| {
            let y = fa + noise.sample(&mut rng);
            StringRecord::from(vec![a.to_string(), y.to_string()])
        })
        .collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        f,
    )
}

fn score(length_scale: f64) -> Result<(), String> {
    let (data, train_truth) = make_dataset(300);
    let xg: Vec<f64> = (0..30).map(|i| 0.02 + 0.96 * (i as f64) / 29.0).collect();
    let probes: Vec<Vec<f64>> = xg.iter().map(|&x| vec![x, 0.0]).collect();
    let probe_truth: Vec<f64> = xg.iter().map(|&x| truth(x)).collect();
    fit_and_score(
        &format!("y ~ matern(x, length_scale={length_scale})"),
        &data,
        &probe_matrix(&probes),
        &probe_truth,
        &train_truth,
    )
}

fn failures(scales: &[f64]) -> Vec<String> {
    scales
        .iter()
        .filter_map(|&ls| score(ls).err().map(|e| format!("ls={ls}: {e}")))
        .collect()
}

#[test]
fn matern_very_small_length_scale_recovers_truth() {
    init_parallelism();
    let failures = failures(&[1e-4, 1e-3, 1e-2]);
    assert!(
        failures.is_empty(),
        "matern small ls failures: {failures:?}"
    );
}

#[test]
fn matern_very_large_length_scale_recovers_truth() {
    init_parallelism();
    let failures = failures(&[10.0, 100.0, 1000.0]);
    assert!(
        failures.is_empty(),
        "matern large ls failures: {failures:?}"
    );
}

#[test]
fn matern_negative_or_zero_length_scale_rejected() {
    init_parallelism();
    let (data, _) = make_dataset(100);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for bad_ls in [-1.0_f64, -0.5, 0.0] {
        let formula = format!("y ~ matern(x, length_scale={bad_ls})");
        match fit_from_formula(&formula, &data, &cfg) {
            Ok(_) => panic!("ls={bad_ls} must be rejected (positive only)"),
            Err(e) => assert_eq!(
                e.error_category(),
                ErrorCategory::Formula,
                "ls={bad_ls} must be refused as an invalid request; got: {e}",
            ),
        }
    }
}
