//! Clamped / anchored B-spline smooths at the minimum-k frontier. Clamped
//! pins f′ = 0 at each end; anchored pins f = f′ = 0 (the Hermite anchor,
//! two constraints per side). Whenever the constrained basis keeps a free
//! coefficient the request is well-posed, so a refusal is a defect and the
//! fit is scored against the known truth by its own posterior band around
//! the basis's best approximation (#4377). Anchored at k=4 leaves the cubic
//! basis no free coefficient at all, so it must be refused as a formula
//! error.

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
use smooth_truth_scoring::{FAMILY_WISE_ALPHA, fit_and_score, probe_matrix};

fn truth(x: f64) -> f64 {
    (std::f64::consts::PI * x).sin()
}

/// Fixture `y = sin(πx) + N(0, 0.05²)`; returns the data and the noise-free
/// truth at the training rows.
fn make_smooth_dataset(n: usize) -> (EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
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

fn failures(bc: &str, ks: &[usize]) -> Vec<String> {
    let (data, train_truth) = make_smooth_dataset(200);
    let xg: Vec<f64> = (0..20).map(|i| 0.02 + 0.96 * (i as f64) / 19.0).collect();
    let probes = probe_matrix(&xg.iter().map(|&x| vec![x, 0.0]).collect::<Vec<_>>());
    let probe_truth: Vec<f64> = xg.iter().map(|&x| truth(x)).collect();
    let alpha = FAMILY_WISE_ALPHA / ks.len() as f64;
    ks.iter()
        .filter_map(|&k| {
            let formula = format!("y ~ s(x, bc={bc}, k={k})");
            fit_and_score(&formula, &data, &probes, &probe_truth, &train_truth, alpha)
                .err()
                .map(|e| format!("{bc} k={k}: {e}"))
        })
        .collect()
}

#[test]
fn bc_clamped_k_min_recovers_truth() {
    init_parallelism();
    let failures = failures("clamped", &[4, 5, 6, 7]);
    assert!(failures.is_empty(), "clamped min-k failures: {failures:?}");
}

#[test]
fn bc_anchored_k_min_recovers_truth() {
    init_parallelism();
    let failures = failures("anchored", &[5, 6, 7, 8]);
    assert!(failures.is_empty(), "anchored min-k failures: {failures:?}");
}

#[test]
fn bc_anchored_both_sides_k4_is_refused_as_a_formula_error() {
    init_parallelism();
    let (data, _) = make_smooth_dataset(200);
    let formula = "y ~ s(x, bc=anchored, k=4)";
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(formula, &data, &cfg) {
        Ok(_) => panic!("{formula} leaves no free coefficient and must be refused"),
        Err(e) => assert_eq!(
            e.error_category(),
            ErrorCategory::Formula,
            "{formula} must be refused as an invalid request; got: {e}",
        ),
    }
}
