//! Sphere smooths must respect sphere topology — the north pole is a single
//! point regardless of longitude, so f(90°, lon) must be identical for all
//! lon. Any smooth that treats `(lat, lon)` as a Euclidean 2D field violates
//! this. Both `sphere(...)` variants (Wahba and harmonic) should pass.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

fn make_dataset(n_lat: usize, n_lon: usize) -> gam::data::EncodedDataset {
    let headers = ["lat", "lon", "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(n_lat * n_lon);
    for i in 0..n_lat {
        // exclude poles to avoid degenerate training points
        let lat = -80.0 + 160.0 * (i as f64) / ((n_lat - 1) as f64);
        for j in 0..n_lon {
            let lon = -180.0 + 360.0 * (j as f64) / (n_lon as f64);
            let lat_r = lat.to_radians();
            let lon_r = lon.to_radians();
            // A smooth signal that genuinely varies with the pole's location
            let y = 0.5
                + 0.6 * lat_r.sin()
                + 0.4 * lat_r.cos() * (2.0 * lon_r).cos()
                + 0.2 * lat_r.cos().powi(2) * lon_r.sin();
            records.push(StringRecord::from(vec![
                lat.to_string(),
                lon.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode sphere dataset")
}

fn predict_data(lats: &[f64], lons: &[f64]) -> Array2<f64> {
    assert_eq!(lats.len(), lons.len());
    let n = lats.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = lats[i];
        m[[i, 1]] = lons[i];
        m[[i, 2]] = 0.0;
    }
    m
}

fn predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    lats: &[f64],
    lons: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("sphere fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let new_data = predict_data(lats, lons);
    let test_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("rebuild design from frozen spec");
    let pred = test_design.design.apply(&fit.fit.beta);
    pred.to_vec()
}

fn assert_pole_invariance(formula: &str, tol: f64) {
    init_parallelism();
    let data = make_dataset(9, 18); // 9 × 18 = 162 samples
    // 12 longitudes at the north pole — all must give the same y.
    let lons: Vec<f64> = (0..12).map(|j| -180.0 + 30.0 * j as f64).collect();
    let lats: Vec<f64> = std::iter::repeat(90.0).take(lons.len()).collect();
    let pred = predict(formula, &data, &lats, &lons);
    let max = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let spread = max - min;
    eprintln!(
        "[sphere-pole] formula=`{formula}` north-pole predictions across 12 lons:\n  \
         min={min:.6} max={max:.6} spread={spread:.6e} (tol={tol:.1e})"
    );
    assert!(
        spread <= tol,
        "north-pole predictions varied by {spread:.6e} across longitudes (tol {tol:.1e}) for formula `{formula}`; \
         a sphere smooth must give a single value at a single point",
    );
}

#[test]
fn sphere_wahba_north_pole_is_a_single_point() {
    assert!(file!().ends_with(".rs"));
    assert_pole_invariance("y ~ sphere(lat, lon, k=8)", 1e-6);
}

#[test]
fn sphere_harmonic_north_pole_is_a_single_point() {
    assert!(file!().ends_with(".rs"));
    assert_pole_invariance("y ~ sphere(lat, lon, method=harmonic, max_degree=4)", 1e-6);
}

/// Sphere smooths must wrap continuously in longitude: f(lat, +180) and
/// f(lat, -180) reference the same point and must give the same prediction.
fn assert_longitude_wrap(formula: &str, tol: f64) {
    init_parallelism();
    let data = make_dataset(9, 18);
    let lats: Vec<f64> = vec![-60.0, -30.0, 0.0, 30.0, 60.0];
    let mut pos_lons = lats.clone();
    pos_lons.iter_mut().for_each(|l| *l = *l * 0.0 + 179.99999);
    let neg_lons: Vec<f64> = pos_lons.iter().map(|_| -179.99999).collect();

    let pred_pos = predict(formula, &data, &lats, &pos_lons);
    let pred_neg = predict(formula, &data, &lats, &neg_lons);

    let mut max_gap = 0.0_f64;
    for (a, b) in pred_pos.iter().zip(pred_neg.iter()) {
        let d = (a - b).abs();
        if d > max_gap {
            max_gap = d;
        }
    }
    eprintln!(
        "[sphere-wrap] formula=`{formula}` max |f(lat, +180) - f(lat, -180)| = {max_gap:.6e} (tol {tol:.1e})"
    );
    assert!(
        max_gap <= tol,
        "longitude wrap discontinuous: max gap {max_gap:.6e} > tol {tol:.1e} for `{formula}`",
    );
}

#[test]
fn sphere_wahba_longitude_wraps_continuously() {
    assert!(file!().ends_with(".rs"));
    assert_longitude_wrap("y ~ sphere(lat, lon, k=8)", 1e-4);
}

#[test]
fn sphere_harmonic_longitude_wraps_continuously() {
    assert!(file!().ends_with(".rs"));
    assert_longitude_wrap("y ~ sphere(lat, lon, method=harmonic, max_degree=4)", 1e-4);
}
