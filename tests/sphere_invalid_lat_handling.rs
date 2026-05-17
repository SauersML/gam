//! Edge cases for sphere lat/lon validation:
//!
//! 1. Training data with lat outside [-90, 90] must produce a clear error.
//! 2. Predict data with lat outside [-90, 90] must produce a clear error,
//!    not propagate NaN/Inf silently.
//! 3. Near-boundary lat = 90.0 (exactly) must succeed.
//! 4. Lat = 90 + ε due to floating-point roundoff (ε = 1e-9) currently fails
//!    the strict validator — flag this as an ergonomics problem if the
//!    failure message is opaque.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

fn make_data_with_lats(lats: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(lats.len() * 4);
    for &lat in lats {
        for j in 0..4 {
            let lon = -90.0 + 90.0 * (j as f64);
            let y = 1.0 + 0.3 * (lat.to_radians()).sin();
            rows.push(StringRecord::from(vec![
                lat.to_string(),
                lon.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn sphere_training_rejects_lat_above_90() {
    init_parallelism();
    let bad_lats: Vec<f64> = (0..40)
        .map(|i| -75.0 + 4.0 * i as f64)
        .chain([95.0])
        .collect();
    let data = make_data_with_lats(&bad_lats);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = fit_from_formula("y ~ sphere(lat, lon, k=10)", &data, &cfg)
        .err()
        .expect("must fail with lat=95");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("latitude") && msg.contains("90"),
        "expected mention of latitude bound, got: {msg}",
    );
}

#[test]
fn sphere_training_rejects_lat_below_neg90() {
    init_parallelism();
    let bad_lats: Vec<f64> = (0..40)
        .map(|i| -75.0 + 4.0 * i as f64)
        .chain([-100.0])
        .collect();
    let data = make_data_with_lats(&bad_lats);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = fit_from_formula("y ~ sphere(lat, lon, k=10)", &data, &cfg)
        .err()
        .expect("must fail with lat=-100");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("latitude") && msg.contains("90"),
        "expected mention of latitude bound, got: {msg}",
    );
}

fn assert_predict_with_invalid_lat_rejects_cleanly(formula: &str) {
    init_parallelism();
    let ok_lats: Vec<f64> = (0..40).map(|i| -75.0 + 4.0 * i as f64).collect();
    let data = make_data_with_lats(&ok_lats);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let mut m = Array2::<f64>::zeros((3, 3));
    m[[0, 0]] = 95.0;
    m[[0, 1]] = 0.0;
    m[[1, 0]] = -91.0;
    m[[1, 1]] = 0.0;
    m[[2, 0]] = 45.0;
    m[[2, 1]] = 0.0;
    let err = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect_err("predict data with latitude outside [-90, 90] must be rejected");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("latitude") || msg.contains("lat"),
        "predict reject without mentioning latitude for `{formula}`: {msg}",
    );
}

#[test]
fn sphere_wahba_predict_with_lat_above_90_rejects_cleanly() {
    assert_predict_with_invalid_lat_rejects_cleanly("y ~ sphere(lat, lon, k=10)");
}

#[test]
fn sphere_harmonic_predict_with_lat_above_90_rejects_cleanly() {
    assert_predict_with_invalid_lat_rejects_cleanly(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
    );
}

#[test]
fn sphere_lat_exactly_at_pole_accepts() {
    init_parallelism();
    let lats: Vec<f64> = (0..20)
        .map(|i| -75.0 + 8.0 * i as f64)
        .chain([90.0, -90.0])
        .collect();
    let data = make_data_with_lats(&lats);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let _ = fit_from_formula("y ~ sphere(lat, lon, k=10)", &data, &cfg)
        .expect("lat=±90 exactly must be accepted as the pole");
}

#[test]
fn sphere_lat_at_90_plus_tiny_eps_fails_with_useful_message() {
    init_parallelism();
    // Floating-point roundoff can push 90.0 to 90.0 + 1e-15. Document the
    // current behavior: strict validator rejects. The error must mention
    // the latitude bound + the offending value so users can diagnose
    // (don't silently clamp — that hides upstream data bugs).
    let lats: Vec<f64> = (0..40)
        .map(|i| -75.0 + 4.0 * i as f64)
        .chain([90.0 + 1e-12])
        .collect();
    let data = make_data_with_lats(&lats);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = fit_from_formula("y ~ sphere(lat, lon, k=10)", &data, &cfg)
        .err()
        .expect("90.0 + 1e-12 should still fail strict validation");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("latitude") && msg.contains("90"),
        "expected actionable lat-bound error, got: {msg}",
    );
}
