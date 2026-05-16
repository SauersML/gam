//! Failing-ticket regression: predicting at the exact training input rows
//! should reproduce the in-sample fitted values to high precision. Any
//! discrepancy indicates a bug in the predict-time design rebuild (it's
//! constructing a different basis from the fit-time design).
//!
//! Setup: 200 sorted x ∈ [0, 1], y = 0.6 sin(2π x) + ε with σ = 0.05.
//! Fit with `s(x)`, then call build_term_collection_design at the same
//! x values. Compare to the in-sample yhat = design.apply(beta). The
//! per-point difference should be ≤ 1e-9.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[test]
fn predict_at_training_points_matches_fitted_values() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(229);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let n = 200usize;
    let x: Vec<f64> = (0..n)
        .map(|i| 0.01 + 0.98 * i as f64 / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| 0.6 * (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=10)", &data, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };

    // In-sample fitted: use the design captured in the fit
    let fitted = fit.design.design.apply(&fit.fit.beta).to_vec();

    // Predict-time rebuild at the same x's
    let mut new_data = Array2::<f64>::zeros((n, 2));
    for (i, &t) in x.iter().enumerate() {
        new_data[[i, 0]] = t;
        new_data[[i, 1]] = 0.0;
    }
    let pred_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("rebuild predict design");
    let pred = pred_design.design.apply(&fit.fit.beta).to_vec();

    assert_eq!(
        fitted.len(),
        pred.len(),
        "fitted and predicted vectors have different length"
    );
    let mut max_diff = 0.0_f64;
    let mut bad_count = 0usize;
    for i in 0..n {
        let d = (fitted[i] - pred[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
        if d > 1e-9 {
            bad_count += 1;
        }
    }
    eprintln!("[predict-train] max_diff={max_diff:.3e} bad_count={bad_count}/{n}");
    assert!(
        max_diff <= 1e-9,
        "predict-at-training mismatched fitted values: max |fit − pred| = {max_diff:.3e} \
         ({bad_count} points exceed 1e-9). This indicates the predict-time design \
         rebuild produces a different basis from the fit-time design."
    );
}
