//! What happens with `s(x, periodic=true)` when the user does NOT specify
//! `period=`? Possible behaviors:
//!   A. Error with actionable message saying period= is required.
//!   B. Infer period from data range (data span).
//!   C. Default to 2π.
//!
//! Any of these is acceptable IF the behavior is documented and consistent.
//! Silent acceptance with unspecified semantics is the bug to guard against.

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

fn make_data_on(range: (f64, f64), period_truth: f64, n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(range.0, range.1).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t.iter().map(|theta| (theta * TAU / period_truth).sin() + noise.sample(&mut rng)).collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t.iter().zip(y.iter())
        .map(|(a,b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn periodic_without_explicit_period_behavior_consistent() {
    init_parallelism();
    let data = make_data_on((0.0, TAU), TAU, 200, 11);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    // Try periodic=true with no period= argument.
    let err = fit_from_formula("y ~ s(t, periodic=true)", &data, &cfg)
        .err()
        .expect(
            "periodic=true without explicit period= must be rejected (silent inference from data range \
             is a user-facing footgun: uniform draws on [0, 2π] give period ≈ 2π − 2ε, not 2π).",
        );
    let lower = err.to_string().to_lowercase();
    assert!(
        lower.contains("period") && lower.contains("explicit"),
        "rejection must name `period` and be actionable; got: {err}",
    );
    eprintln!("[per-default] correctly rejected: {err}");
}

#[test]
fn periodic_with_explicit_period_matches_truth() {
    init_parallelism();
    // Data on [0, 2π], truth period 2π. With explicit period=2π, fit must
    // recover well.
    let data = make_data_on((0.0, TAU), TAU, 300, 11);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &data,
        &cfg,
    ).expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!() };
    let probes: Vec<f64> = (0..50).map(|i| TAU * (i as f64) / 49.0).collect();
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &v) in probes.iter().enumerate() { m[[i, 0]] = v; m[[i, 1]] = 0.0; }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let truth: Vec<f64> = probes.iter().map(|t| t.sin()).collect();
    let sumsq: f64 = pred.iter().zip(truth.iter()).map(|(p, t)| (p - t).powi(2)).sum();
    let rmse = (sumsq / pred.len() as f64).sqrt();
    eprintln!("[per-explicit] rmse vs truth = {rmse:.4}");
    assert!(rmse < 0.1, "explicit period=2π fit should recover sin(t): rmse={rmse:.4}");
}
