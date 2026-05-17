//! Predicting at latitudes very close to the pole (`lat = 89.99..., 90`)
//! should produce a smoothly-varying result, not blow up. The kernel
//! evaluates `(1 - cos γ).max(EPS·1e-4)` to keep `log(u)` finite, but
//! we want to verify the predicted *function* is also continuous.

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

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.6 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict_at(formula: &str, lats: &[f64], lons: &[f64]) -> Vec<f64> {
    let data = make_dataset(400);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = lats.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = lats[i];
        m[[i, 1]] = lons[i];
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn sphere_sobolev_predict_continuous_as_lat_approaches_pole() {
    init_parallelism();
    let lats = vec![89.0, 89.5, 89.9, 89.99, 89.999, 89.9999, 90.0];
    let lons = vec![0.0; lats.len()];
    let pred = predict_at("y ~ sphere(lat, lon, k=30, kernel=sobolev)", &lats, &lons);
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite near pole");
    eprintln!("[near-pole-sob] preds: {pred:?}");
    // Consecutive predictions should differ smoothly (no jump at the
    // pole-floor activation).
    for i in 1..pred.len() {
        let jump = (pred[i] - pred[i - 1]).abs();
        assert!(
            jump < 0.1,
            "sphere fit jumps at lat={}: pred[{}]={:.6} vs pred[{}]={:.6} (jump {jump:.4})",
            lats[i],
            i - 1,
            pred[i - 1],
            i,
            pred[i],
        );
    }
}

#[test]
fn sphere_pseudo_predict_continuous_as_lat_approaches_pole() {
    init_parallelism();
    let lats = vec![89.0, 89.5, 89.9, 89.99, 89.999, 89.9999, 90.0];
    let lons = vec![0.0; lats.len()];
    let pred = predict_at("y ~ sphere(lat, lon, k=30, kernel=pseudo)", &lats, &lons);
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite near pole");
    eprintln!("[near-pole-pse] preds: {pred:?}");
    for i in 1..pred.len() {
        let jump = (pred[i] - pred[i - 1]).abs();
        assert!(
            jump < 0.1,
            "sphere fit jumps at lat={}: jump={jump:.4}",
            lats[i],
        );
    }
}

#[test]
fn sphere_predict_at_exact_pole_is_lon_independent() {
    // At lat = ±90 exactly, the longitude is meaningless. Predictions
    // at (90°, lon₁) and (90°, lon₂) must be identical.
    init_parallelism();
    let lats = vec![90.0, 90.0, 90.0, 90.0, -90.0, -90.0, -90.0];
    let lons = vec![-180.0, -90.0, 0.0, 90.0, -45.0, 0.0, 135.0];
    for kernel in ["sobolev", "pseudo"] {
        let pred = predict_at(
            &format!("y ~ sphere(lat, lon, k=30, kernel={kernel})"),
            &lats,
            &lons,
        );
        // First 4 (north pole) should all equal each other; last 3 should match each other.
        let north_max = pred[..4].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let north_min = pred[..4].iter().cloned().fold(f64::INFINITY, f64::min);
        let south_max = pred[4..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let south_min = pred[4..].iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "[pole-lon-inv {kernel}] north range [{north_min:.6}, {north_max:.6}], south [{south_min:.6}, {south_max:.6}]"
        );
        assert!(
            (north_max - north_min).abs() < 1e-9,
            "[{kernel}] north pole prediction depends on lon (range {})",
            north_max - north_min,
        );
        assert!(
            (south_max - south_min).abs() < 1e-9,
            "[{kernel}] south pole prediction depends on lon (range {})",
            south_max - south_min,
        );
    }
}
